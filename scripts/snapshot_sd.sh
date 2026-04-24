#!/bin/bash
# snapshot_sd.sh — Capture a clean, distributable image of a configured SD card.
#
# Pipeline:
#   1. Shred the tmpfs mountpoint contents (BDD/, .git, models) so the
#      sensitive bytes are overwritten in place before the blocks are freed.
#   2. Delete /var/log contents (auth.log, apt history, journald, etc.) so
#      they don't reveal network IPs, installed packages, or login activity.
#   3. Delete per-user history files (shell/REPL/editor/DB client) under
#      /root and /home/*.
#   4. Delete VSCode / Cursor / Codium remote-server state under /root and
#      /home/* (edit history, workspace state, extension data).
#   5. Zero-fill the remaining free space on rootfs so other stale data in
#      previously-freed blocks does not leak into the image.
#   6. dd the whole device to an image file.
#   7. Shrink the image with pishrink so it fits on any card at least as
#      large as the shrunk filesystem.
#
# The output image is safe to flash to a fresh SD card with setup_sd.sh
# (or plain dd) without leaking contents of the sensitive directories.
# Note: this workflow does NOT sanitize the SOURCE card's unmapped NAND
# cells — only the produced image. Treat the source as still hot.
#
# Usage: sudo ./snapshot_sd.sh [OPTIONS] <device> <image>
#
#   device    Block device of the source SD  (e.g. /dev/sdb, /dev/mmcblk0)
#   image     Output image path              (e.g. ./bdd-clean.img)
#
# Options:
#   --keep-sensitive   Skip step 1 (leave BDD/, .git, models intact)
#   --keep-logs        Skip step 2 (leave /var/log contents)
#   --keep-history     Skip step 3 (leave shell/editor/REPL histories)
#   --keep-vscode      Skip step 4 (leave VSCode/Cursor remote-server state)
#   --no-zerofill      Skip step 5 (faster but may leave stale free-block data)
#   --no-shrink        Skip step 7 (produces a full device-size image)
#   --force            Don't prompt before modifying the source card
#   -h, --help         Show this help

set -euo pipefail

# -- Globals -------------------------------------------------------------------
DEVICE=""
IMAGE=""
KEEP_SENSITIVE=false
KEEP_LOGS=false
KEEP_HISTORY=false
KEEP_VSCODE=false
NO_ZEROFILL=false
NO_SHRINK=false
FORCE=false

ROOTFS=""
BOOT_PART=""
ROOT_PART=""

# Paths (relative to rootfs root) that configure_tmpfs uses as tmpfs mount
# points. Keep in sync with setup_sd.sh:configure_tmpfs.
SENSITIVE_DIRS=(
    "home/bdd/hailo-rpi5-examples/BDD"
    "home/bdd/hailo-rpi5-examples/.git"
    "home/bdd/hailo-rpi5-examples/scripts"
    "home/bdd/models"
)

# Per-user history files (shell, REPL, editor, DB clients, debugger). Each
# one records interactive activity. Names are relative to a home directory;
# wipe_history iterates over /root and every /home/* entry.
HISTORY_FILES=(
    .bash_history
    .zsh_history
    .ash_history
    .sh_history
    .python_history
    .node_repl_history
    .lesshst
    .viminfo
    .mysql_history
    .psql_history
    .sqlite_history
    .gdb_history
)

# VSCode-family remote-server directories. The .vscode-server tree in
# particular holds User/History/* (timestamped snapshots of every edited
# file — leaks BDD/ contents even though BDD/ itself is tmpfs), plus logs,
# workspace state, and installed extensions.
VSCODE_DIRS=(
    .vscode-server
    .vscode-server-insiders
    .vscode-remote
    .vscode
    .cursor-server
    .codium-server
)

# -- Helpers -------------------------------------------------------------------
die()  { echo "ABORT: $*" >&2; exit 1; }
step() { echo "==> $*"; }

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
    exit 1
}

# Catch silent set -e failures — print the line, command, and exit code so
# a failed check or missing dependency can't make the script look like a no-op.
on_err() {
    local rc=$?
    echo "ABORT: exited with status $rc at line $1" >&2
    echo "  command: ${BASH_COMMAND}" >&2
    exit "$rc"
}
trap 'on_err $LINENO' ERR

# -- Safety checks -------------------------------------------------------------
safety_checks() {
    local dev="$1"
    step "Running safety checks on $dev"

    # Size guard — no consumer SD exceeds 512 GiB. Also catches the common
    # case where the card isn't readable (blockdev returns 0 / errors).
    local size_bytes size_gib
    size_bytes=$(sudo blockdev --getsize64 "$dev" 2>/dev/null || echo "0")
    size_gib=$(( size_bytes / 1024 / 1024 / 1024 ))
    echo "    Size: ${size_gib} GiB (${size_bytes} bytes)"
    [[ $size_bytes -eq 0 ]] && \
        die "blockdev --getsize64 $dev returned 0 — card missing or unreadable?"
    [[ $size_gib -gt 512 ]] && \
        die "$dev is ${size_gib} GiB — too large to be an SD card (limit: 512 GiB)."

    # Reject if any partition serves a live OS path.
    echo "    Checking for live system mounts on $dev..."
    local critical_paths=("/" "/boot" "/boot/efi" "/boot/firmware" "/home" "/usr" "/var" "/etc")
    local src mnt path
    while IFS=' ' read -r src mnt _; do
        [[ "$src" != "${dev}"* ]] && continue
        for path in "${critical_paths[@]}"; do
            [[ "$mnt" == "$path" ]] && \
                die "${src} is mounted at '${mnt}' — refusing to image a live system disk."
        done
    done < /proc/mounts

    # Reject if this is the disk backing the running root.
    echo "    Checking whether $dev backs the running OS..."
    local running_root_src running_root_disk
    running_root_src=$(findmnt -n -o SOURCE / 2>/dev/null || true)
    if [[ -n "$running_root_src" ]]; then
        running_root_disk=$(lsblk -no pkname "$running_root_src" 2>/dev/null || true)
        running_root_disk="${running_root_disk:-$running_root_src}"
        [[ "$running_root_disk" != /dev/* ]] && running_root_disk="/dev/${running_root_disk}"
        [[ "$running_root_disk" == "$dev" ]] && \
            die "$dev backs the currently running OS root."
        echo "    Running root is on $running_root_disk — OK"
    fi

    step "Safety checks passed"
}

# -- Unmount any auto-mounted partitions of the source -------------------------
unmount_auto_mounts() {
    local dev="$1"
    local src _rest
    # Unmount in reverse to avoid nested-mount ordering issues.
    local to_unmount=()
    while IFS=' ' read -r src _rest; do
        [[ "$src" == "${dev}"* ]] && to_unmount+=("$src")
    done < /proc/mounts
    local i
    for (( i=${#to_unmount[@]}-1; i>=0; i-- )); do
        step "Unmounting auto-mount ${to_unmount[$i]}"
        sudo umount "${to_unmount[$i]}" || die "Failed to unmount ${to_unmount[$i]} — close any apps using it."
    done
}

# -- Partition detection + rootfs mount ----------------------------------------
mount_rootfs() {
    local dev="$1"
    sudo partprobe "$dev" 2>/dev/null || true
    sleep 1

    if [[ "$dev" =~ (mmcblk|nvme) ]]; then
        BOOT_PART="${dev}p1"
        ROOT_PART="${dev}p2"
    else
        BOOT_PART="${dev}1"
        ROOT_PART="${dev}2"
    fi
    [[ -b "$ROOT_PART" ]] || die "Root partition $ROOT_PART not found."

    step "Mounting rootfs ($ROOT_PART) at $ROOTFS"
    sudo mount "$ROOT_PART" "$ROOTFS" || die "Could not mount $ROOT_PART"

    # Sanity: does this look like an RPi rootfs?
    [[ -f "$ROOTFS/etc/fstab" ]] || die "$ROOT_PART does not look like an RPi rootfs (no /etc/fstab)."
}

# -- Step 1: shred and remove sensitive directories ----------------------------
wipe_sensitive() {
    step "Shredding sensitive directories..."
    local d path
    for d in "${SENSITIVE_DIRS[@]}"; do
        path="$ROOTFS/$d"
        if [[ -d "$path" ]]; then
            echo "    /$d"
            # Overwrite file data blocks in place so freed blocks are zeros.
            sudo find "$path" -mindepth 1 -type f -exec shred -f -n 1 -z -u {} + 2>/dev/null || true
            sudo find "$path" -mindepth 1 -depth -delete
        else
            echo "    /$d  (not present, skipped)"
        fi
    done
    sync
}

# -- Step 2: delete logs -------------------------------------------------------
# Remove regular files under /var/log (system logs, journald, apt history,
# auth.log, wtmp/btmp/lastlog) plus any pre-existing coredumps that could
# hold in-memory copies of sensitive data. Directory structure is preserved
# so daemons can recreate their own files on next boot. Zero-fill below
# scrubs the freed blocks.
wipe_logs() {
    if [[ -d "$ROOTFS/var/log" ]]; then
        step "Removing /var/log contents..."
        sudo find "$ROOTFS/var/log" -type f -delete
    else
        step "No /var/log on rootfs — skipping log wipe."
    fi

    local coredump_path
    for coredump_path in "$ROOTFS/var/lib/systemd/coredump" "$ROOTFS/var/crash"; do
        if [[ -d "$coredump_path" ]]; then
            step "Removing coredumps under ${coredump_path#"$ROOTFS"}..."
            sudo find "$coredump_path" -mindepth 1 -type f -delete
        fi
    done
    sync
}

# -- Step 3: delete per-user history files -------------------------------------
# For /root and each /home/<user>, remove well-known history/state files that
# record interactive activity. Uses plain rm; the zero-fill step scrubs the
# freed blocks.
wipe_history() {
    step "Removing shell/editor/REPL history files..."
    local home_dirs=("$ROOTFS/root")
    local d
    if [[ -d "$ROOTFS/home" ]]; then
        for d in "$ROOTFS/home"/*/; do
            [[ -d "$d" ]] && home_dirs+=("${d%/}")
        done
    fi

    local home f removed=0
    for home in "${home_dirs[@]}"; do
        [[ -d "$home" ]] || continue
        for f in "${HISTORY_FILES[@]}"; do
            if [[ -e "$home/$f" ]]; then
                sudo rm -f "$home/$f"
                echo "    ${home#"$ROOTFS"}/$f"
                removed=$((removed + 1))
            fi
        done
    done
    echo "    ($removed file(s) removed)"
    sync
}

# -- Step 4: delete VSCode / Cursor / Codium remote-server state ---------------
# VSCode Remote-SSH installs ~/.vscode-server/ on the remote host containing
# User/History/* (edit snapshots of every file opened), workspace state,
# extension data, and logs. These leak contents of tmpfs-backed files onto
# persistent ext4. Wipe the whole tree; zero-fill below scrubs freed blocks.
wipe_vscode() {
    step "Removing VSCode / Cursor / Codium remote-server state..."
    local home_dirs=("$ROOTFS/root")
    local d
    if [[ -d "$ROOTFS/home" ]]; then
        for d in "$ROOTFS/home"/*/; do
            [[ -d "$d" ]] && home_dirs+=("${d%/}")
        done
    fi

    local home name path removed=0
    for home in "${home_dirs[@]}"; do
        [[ -d "$home" ]] || continue
        for name in "${VSCODE_DIRS[@]}"; do
            path="$home/$name"
            if [[ -e "$path" ]]; then
                sudo rm -rf "$path"
                echo "    ${path#"$ROOTFS"}"
                removed=$((removed + 1))
            fi
        done
    done
    echo "    ($removed item(s) removed)"
    sync
}

# -- Step 5: zero-fill remaining free space ------------------------------------
zerofill() {
    local zerofile="$ROOTFS/ZEROFILL.tmp"
    step "Zero-filling free space on rootfs (writes until disk full — expected)..."
    # dd exits 1 when the filesystem fills, which is the termination condition.
    sudo dd if=/dev/zero of="$zerofile" bs=4M status=progress 2>&1 || true
    sync
    sudo rm -f "$zerofile"
    sync
}

# -- Step 6: dd the device to an image -----------------------------------------
snapshot_device() {
    local dev="$1"
    local image="$2"
    step "Imaging $dev to $image ..."
    sudo blockdev --flushbufs "$dev" 2>/dev/null || true
    sudo dd if="$dev" of="$image" bs=4M status=progress conv=fsync
    sync
}

# -- Step 7: shrink image with pishrink ----------------------------------------
shrink_image() {
    local image="$1"
    local pishrink_cmd=""
    if command -v pishrink.sh >/dev/null 2>&1; then
        pishrink_cmd=$(command -v pishrink.sh)
    elif command -v pishrink >/dev/null 2>&1; then
        pishrink_cmd=$(command -v pishrink)
    else
        die "pishrink not found on PATH.
   Install from https://github.com/Drewsif/PiShrink, or re-run with --no-shrink."
    fi

    step "Shrinking $image with $pishrink_cmd ..."
    sudo "$pishrink_cmd" -rn "$image"
}

# -- Cleanup (EXIT trap) -------------------------------------------------------
cleanup() {
    sync 2>/dev/null || true
    [[ -n "$ROOTFS" ]] && sudo umount "$ROOTFS" 2>/dev/null || true
    [[ -n "$ROOTFS" ]] && rmdir "$ROOTFS" 2>/dev/null || true
}


# -- Arg parsing ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-sensitive) KEEP_SENSITIVE=true; shift ;;
        --keep-logs)      KEEP_LOGS=true;      shift ;;
        --keep-history)   KEEP_HISTORY=true;   shift ;;
        --keep-vscode)    KEEP_VSCODE=true;    shift ;;
        --no-zerofill)    NO_ZEROFILL=true;    shift ;;
        --no-shrink)      NO_SHRINK=true;      shift ;;
        --force)          FORCE=true;          shift ;;
        -h|--help)        usage ;;
        -*)               echo "Unknown option: $1"; usage ;;
        *)
            if   [[ -z "$DEVICE" ]]; then DEVICE="$1"
            elif [[ -z "$IMAGE"  ]]; then IMAGE="$1"
            else echo "Unexpected argument: $1"; usage
            fi
            shift
            ;;
    esac
done

[[ -z "$DEVICE" ]]   && { echo "Error: device not specified."; usage; }
[[ -z "$IMAGE"  ]]   && { echo "Error: image path not specified."; usage; }
[[ ! -b "$DEVICE" ]] && die "'$DEVICE' is not a block device."
[[ -e "$IMAGE"  ]]   && die "'$IMAGE' already exists — remove it or choose another path."

# Image destination must be writable by the user who'll hold the file. Check
# parent dir now so we don't dd to completion and then fail on rename/move.
IMAGE_DIR=$(dirname "$IMAGE")
[[ -d "$IMAGE_DIR" ]] || die "Image directory '$IMAGE_DIR' does not exist."
[[ -w "$IMAGE_DIR" ]] || die "Image directory '$IMAGE_DIR' is not writable by $USER."

# -- Main ----------------------------------------------------------------------
step "snapshot_sd.sh starting"
echo "    DEVICE : $DEVICE"
echo "    IMAGE  : $IMAGE"
echo "    Flags  : keep-sensitive=$KEEP_SENSITIVE  keep-logs=$KEEP_LOGS  keep-history=$KEEP_HISTORY"
echo "             keep-vscode=$KEEP_VSCODE  no-zerofill=$NO_ZEROFILL  no-shrink=$NO_SHRINK  force=$FORCE"

# Verify sudo access up front. Otherwise a later sudo prompt could time out
# silently and set -e would trip without any visible context.
if [[ $EUID -ne 0 ]]; then
    step "Verifying sudo access (may prompt for password)..."
    sudo -v || die "sudo access required to read/modify $DEVICE."
fi

# Fail fast if pishrink is required but missing, so we don't spend 10 minutes
# on dd only to error out at the last step.
if ! $NO_SHRINK; then
    if ! command -v pishrink.sh >/dev/null 2>&1 \
       && ! command -v pishrink >/dev/null 2>&1; then
        die "pishrink not found on PATH.
   Install from https://github.com/Drewsif/PiShrink, or re-run with --no-shrink."
    fi
fi

safety_checks "$DEVICE"

# Confirm before we touch the source card (wipe modifies it in place).
if ! $KEEP_SENSITIVE && ! $FORCE; then
    echo "-- Source device ------------------------------------------------------------"
    lsblk -o NAME,SIZE,TYPE,TRAN,VENDOR,MODEL,MOUNTPOINTS "$DEVICE" 2>/dev/null \
        || lsblk "$DEVICE"
    echo "-----------------------------------------------------------------------------"
    echo "This will SHRED and REMOVE the following directories on $DEVICE:"
    for d in "${SENSITIVE_DIRS[@]}"; do echo "  /$d"; done
    read -rp "Type 'yes' to continue: " confirm
    [[ "$confirm" != "yes" ]] && { echo "Aborted."; exit 0; }
fi

readonly ROOTFS=$(mktemp -d /tmp/rpi-rootfs-XXXXXX)
trap cleanup EXIT

unmount_auto_mounts "$DEVICE"
mount_rootfs        "$DEVICE"

if ! $KEEP_SENSITIVE; then
    wipe_sensitive
else
    step "Skipping sensitive-dir wipe (--keep-sensitive)."
fi

if ! $KEEP_LOGS; then
    wipe_logs
else
    step "Skipping log wipe (--keep-logs)."
fi

if ! $KEEP_HISTORY; then
    wipe_history
else
    step "Skipping history wipe (--keep-history)."
fi

if ! $KEEP_VSCODE; then
    wipe_vscode
else
    step "Skipping VSCode wipe (--keep-vscode)."
fi

if ! $NO_ZEROFILL; then
    zerofill
else
    step "Skipping zero-fill (--no-zerofill)."
fi

sync
sudo umount "$ROOTFS"
rmdir "$ROOTFS"
# cleanup()'s umount/rmdir tolerate this already-unmounted state via '|| true'.

snapshot_device "$DEVICE" "$IMAGE"

if ! $NO_SHRINK; then
    shrink_image "$IMAGE"
else
    step "Skipping shrink (--no-shrink)."
fi

trap - EXIT

echo ""
echo "Done."
echo "  Source : $DEVICE"
echo "  Image  : $IMAGE ($(du -h "$IMAGE" | cut -f1))"
echo ""
echo "Flash the image onto a fresh card with:"
echo "  sudo ./setup_sd.sh --image $IMAGE <new-device>"
