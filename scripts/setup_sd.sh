#!/bin/bash
# setup_sd.sh — Flash and configure a Raspberry Pi OS SD card.
#
# Usage: sudo ./setup_sd.sh [OPTIONS] <device>
#
#   device           Block device of the SD card (e.g. /dev/sdb, /dev/mmcblk0)
#
# Options:
#   --image PATH     RPi OS image to flash (.img, .img.xz, .img.gz)
#   --ip ADDR/CIDR   Static IP for eth0     (default: 192.168.1.100/24)
#   --gateway GW     Default gateway        (default: 192.168.0.1)
#   --no-perf        Disable performance optimizations in config.txt
#   --force          Skip removable-device check (use with care)
#   --only=FUNC      Run only one setup function (device is still required for
#                    mounting). FUNC is one of:
#                      flash_sd           — safety checks + flash + mount
#                      configure_boot     — cmdline.txt + config.txt
#                      configure_user     — userconf.txt (bdd user)
#                      configure_ssh      — SSH enable, authorized_keys, sshd config
#                      configure_network  — static IP
#                      configure_wifi     — WiFi (wpa_supplicant.conf)
#                      configure_px4      — PX4 autostart systemd service
#
# Credentials written to SD:
#   User: bdd   Password: 1111
#   SSH public key for v.nemkov@gmail.com

set -euo pipefail

# -- Globals -------------------------------------------------------------------
PERF_OPTS=true
FORCE=false
IMAGE=""
DEVICE=""
STATIC_IP="192.168.1.100/24"
GATEWAY="192.168.0.1"
ONLY=""

# bdd user, password 1111 (SHA-512 hash)
USERCONF_LINE='bdd:$6$qWNVYvlyVW.0jAum$Qq1YrOptv8jzz./v/VC5ttVwiQP7r.dtBA5cbq/5GY9zEc9YUFo/xrtY5GKAV9/PDuxiIOPofwdO3eOAoX.G9.'
SSH_PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE17TRhHsjBCNssO94hQjWQmK/9UYz6Bx9WJfFLJ2gqP v.nemkov@gmail.com"

BOOTFS=""
ROOTFS=""
BOOT_PART=""
ROOT_PART=""

# -- Helpers -------------------------------------------------------------------
die()  { echo "ABORT: $*" >&2; exit 1; }
step() { echo "==> $*"; }

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
    exit 1
}

# -- Safety checks and flash ---------------------------------------------------
# Validates the target device, shows device info, asks for confirmation,
# then flashes the image if --image was supplied.
flash_sd() {
    local dev="$1"
    local image="$2"
    local dev_name
    dev_name=$(basename "$dev")

    # Doesn't work on my machine -- sd cards are reported non-removable
    # # 1. Removable flag — SD cards via card readers always report 1.
    # local removable
    # removable=$(cat "/sys/block/${dev_name}/removable" 2>/dev/null || echo "unknown")
    # if [[ "$removable" != "1" ]]; then
    #     if $FORCE; then
    #         echo "WARNING: $dev is not marked removable (removable=$removable) — continuing due to --force."
    #     else
    #         die "$dev is not marked as a removable device (removable=$removable).
    #    This guard prevents accidentally flashing a system disk.
    #    Re-run with --force if you are certain this is an SD card."
    #     fi
    # fi

    # 2. Size guard — no consumer SD card exceeds 512 GiB.
    local size_bytes size_gib
    size_bytes=$(sudo blockdev --getsize64 "$dev" 2>/dev/null || echo "0")
    size_gib=$(( size_bytes / 1024 / 1024 / 1024 ))
    [[ $size_gib -gt 512 ]] && \
        die "$dev is ${size_gib} GiB — too large to be an SD card (limit: 512 GiB)."

    # 3. Critical mount check — refuse if any partition serves a live OS path.
    local critical_paths=("/" "/boot" "/boot/efi" "/boot/firmware" "/home" "/usr" "/var" "/etc")
    local src mnt
    while IFS=' ' read -r src mnt _; do
        [[ "$src" != "${dev}"* ]] && continue
        local path
        for path in "${critical_paths[@]}"; do
            [[ "$mnt" == "$path" ]] && \
                die "${src} is mounted at '${mnt}' — refusing to flash a device with live system mounts."
        done
    done < /proc/mounts

    # 4. OS boot disk check — compare against the disk backing /.
    local running_root_src running_root_disk
    running_root_src=$(findmnt -n -o SOURCE / 2>/dev/null || true)
    if [[ -n "$running_root_src" ]]; then
        running_root_disk=$(lsblk -no pkname "$running_root_src" 2>/dev/null || true)
        running_root_disk="${running_root_disk:-$running_root_src}"
        [[ "$running_root_disk" != /dev/* ]] && running_root_disk="/dev/${running_root_disk}"
        [[ "$running_root_disk" == "$dev" ]] && \
            die "$dev is the disk that holds the currently running OS root filesystem."
    fi

    if [[ -n "$image" ]]; then
        # 5. Show device info for visual confirmation.
        echo "-- Target device ------------------------------------------------------------"
        lsblk -o NAME,SIZE,TYPE,TRAN,VENDOR,MODEL,MOUNTPOINTS "$dev" 2>/dev/null \
            || lsblk "$dev"
        echo "-----------------------------------------------------------------------------"
        echo "WARNING: This will OVERWRITE all data on $dev"
        read -rp "Type 'yes' to continue: " confirm
        [[ "$confirm" != "yes" ]] && { echo "Aborted."; exit 0; }

        step "Flashing $image to $dev ..."
        case "$image" in
            *.xz) xzcat "$image" | sudo dd of="$dev" bs=4M status=progress conv=fsync ;;
            *.gz) zcat  "$image" | sudo dd of="$dev" bs=4M status=progress conv=fsync ;;
            *)    sudo dd if="$image" of="$dev" bs=4M status=progress conv=fsync ;;
        esac
        sync
        step "Flash complete."
    else
        step "No --image specified; skipping flash (assuming $dev already has RPi OS)."
    fi

    # 7. Detect partitions and mount.
    mount_partitions "$dev"
}

# -- Partition detection and mount ---------------------------------------------
mount_partitions() {
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
    step "Boot partition : $BOOT_PART"
    step "Root partition : $ROOT_PART"

    sudo mount "$BOOT_PART" "$BOOTFS" ||:
    sudo mount "$ROOT_PART" "$ROOTFS" ||:
}

# -- Boot config (cmdline.txt + config.txt) ------------------------------------
configure_boot() {
    local bootfs="$1"
    local perf_opts="$2"

    # Extract PARTUUID from the image's own cmdline.txt — must not be hardcoded.
    local cmdline_src="$bootfs/cmdline.txt"
    [[ ! -f "$cmdline_src" ]] && \
        die "$cmdline_src not found — is this a valid RPi OS image?"
    local partuuid
    partuuid=$(grep -oP 'root=PARTUUID=\K\S+' "$cmdline_src")
    [[ -z "$partuuid" ]] && \
        die "Could not extract PARTUUID from $cmdline_src (content: $(cat "$cmdline_src"))"
    step "PARTUUID: $partuuid"

    # cmdline.txt — must be a single line.
    step "Writing cmdline.txt..."
    printf 'console=tty1 root=PARTUUID=%s rootfstype=ext4 fsck.repair=yes rootwait elevator=deadline isolcpus=2 quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=PL quiet\n' \
        "$partuuid" | sudo tee "$bootfs/cmdline.txt" > /dev/null

    # config.txt — main body.
    step "Writing config.txt..."
    sudo tee "$bootfs/config.txt" > /dev/null <<'ENDMAIN'
# For more options and information see http://rptl.io/configtxt

# Enable audio (loads snd_bcm2835)
dtparam=audio=on

# Additional overlays documented in /boot/firmware/overlays/README

# Enable sc16is752 over SPI1
dtoverlay=sc16is752-spi1
# Enable I2C-1 at 400 kHz
dtparam=i2c_arm=on,i2c_arm_baudrate=400000
# Enable SPI (spidev0.0)
dtparam=spi=on
# Enable full UART0 for RC input; disable Bluetooth to free it
enable_uart=1
dtoverlay=uart0
dtoverlay=disable-bt
# Enable I2C-0
dtparam=i2c_vc=on

# -- Cameras ------------------------------------------------------------------
# Disable firmware auto-detection; overlays configured explicitly below.
camera_auto_detect=0
#dtoverlay=arducam_64mp,cam0
dtoverlay=ov5647,cam0
#dtoverlay=imx477,cam0
#dtoverlay=imx477,cam1
dtoverlay=ov5647,cam1
#dtoverlay=arducam_64mp,cam1
dtoverlay=arducam-64mp,cam0

# -- Display / framebuffer -----------------------------------------------------
display_auto_detect=1
auto_initramfs=1
dtoverlay=vc4-kms-v3d
max_framebuffers=2
# Don't let firmware inject video= into cmdline.txt
disable_fw_kms_setup=1

# -- CPU / memory --------------------------------------------------------------
arm_64bit=1
disable_overscan=1
gpu_mem=512
# Clock boost allowed by firmware/thermal headroom
arm_boost=1

# -- PCIe ----------------------------------------------------------------------
# Run PCIe at Gen 3 full speed (RPi 5 / CM5)
dtparam=pciex1_gen=3
ENDMAIN

    # config.txt — optional performance block.
    if $perf_opts; then
        sudo tee -a "$bootfs/config.txt" > /dev/null <<'ENDPERF'

# -- Performance optimizations (disable with --no-perf) ------------------------
# Hold CPU at maximum frequency — prevents latency spikes from frequency scaling.
force_turbo=1
# Small voltage increase for RPi 5 stability at full clock.
over_voltage_delta=50000
ENDPERF
    else
        printf '\n# Performance optimizations disabled (--no-perf)\n' \
            | sudo tee -a "$bootfs/config.txt" > /dev/null
    fi

    # config.txt — board-specific sections.
    sudo tee -a "$bootfs/config.txt" > /dev/null <<'ENDBOARDS'

[cm4]
# Enable host-mode XHCI USB on CM4
otg_mode=1

[cm5]
dtoverlay=dwc2,dr_mode=host
ENDBOARDS
}

# -- User setup (userconf + SSH key) ------------------------------------------
configure_user() {
    local bootfs="$1"
    local rootfs="$2"

    # userconf.txt — creates user 'bdd' with password '1111' on first boot.
    step "Writing userconf.txt (user: bdd / password: 1111)..."
    printf '%s\n' "$USERCONF_LINE" | sudo tee "$bootfs/userconf.txt" > /dev/null
}

configure_ssh() {
    # Enable SSH.
    step "Enabling SSH..."
    sudo touch "$bootfs/ssh"

    # SSH authorized_keys — collect all public keys from the host's ~/.ssh/*.pub
    # and write them to the SD card. Fall back to the hardcoded key if none found.
    # First user on RPi OS is always UID/GID 1000.
    local bdd_ssh="$rootfs/home/bdd/.ssh"
    sudo mkdir -p "$bdd_ssh"

    local host_ssh_dir
    host_ssh_dir=$(eval echo "~${SUDO_USER:-$USER}/.ssh")
    local -a pub_keys
    mapfile -t pub_keys < <(cat "$host_ssh_dir"/*.pub 2>/dev/null)

    if [[ ${#pub_keys[@]} -gt 0 ]]; then
        step "Installing ${#pub_keys[@]} SSH public key(s) from $host_ssh_dir..."
        printf '%s\n' "${pub_keys[@]}" | sudo tee "$bdd_ssh/authorized_keys" > /dev/null
    else
        step "No public keys found in $host_ssh_dir — falling back to hardcoded key..."
        printf '%s\n' "$SSH_PUBKEY" | sudo tee "$bdd_ssh/authorized_keys" > /dev/null
    fi

    sudo chown -R 1000:1000 "$rootfs/home/bdd"
    sudo chmod 700 "$bdd_ssh"
    sudo chmod 600 "$bdd_ssh/authorized_keys"

    # Allow password-based SSH login.
    # RPi OS ships with PasswordAuthentication no; override via drop-in.
    step "Enabling password SSH authentication..."
    sudo mkdir -p "$rootfs/etc/ssh/sshd_config.d"
    sudo tee "$rootfs/etc/ssh/sshd_config.d/10-allow-password.conf" > /dev/null <<'ENDSSH'
PasswordAuthentication yes
ENDSSH
}

# -- Network config (static IP + WiFi) ----------------------------------------
configure_network() {
    local rootfs="$1"
    local static_ip="$2"
    local gateway="$3"

    step "Configuring static IP: $static_ip, gateway: $gateway ..."
    sudo mkdir -p "$rootfs/etc/network/interfaces.d"
    sudo tee "$rootfs/etc/network/interfaces.d/eth0" > /dev/null <<ENDETH
allow-hotplug eth0
iface eth0 inet static
address ${static_ip}
netmask 255.255.255.0
gateway ${gateway}
ENDETH
}

# -- WiFi config ---------------------------------------------------------------
# Reads up to 10 most-recently-used WiFi networks from NetworkManager on the
# host, presents a numbered selection menu, then writes a wpa_supplicant.conf
# to the SD rootfs for the chosen networks.
configure_wifi() {
    local rootfs="$1"

    # Collect WiFi connections sorted by timestamp descending, newest first.
    # nmcli outputs: TIMESTAMP:NAME  (tab-separated when -t is used)
    local -a names timestamps
    while IFS=: read -r _type ts name; do
        names+=("$name")
        timestamps+=("$ts")
    done < <(nmcli -t -f TYPE,TIMESTAMP,NAME connection show \
                | grep '^802-11-wireless:' \
                | sort -t: -k2 -rn \
                | head -10)

    if [[ ${#names[@]} -eq 0 ]]; then
        echo "  No WiFi networks found in NetworkManager — skipping WiFi config."
        return
    fi

    # Show selection menu.
    echo ""
    echo "-- WiFi networks (most recent first) ----------------------------------------"
    local i
    for (( i=0; i<${#names[@]}; i++ )); do
        # Convert epoch timestamp to human-readable date.
        local ts_human
        ts_human=$(date -d "@${timestamps[$i]}" '+%Y-%m-%d %H:%M' 2>/dev/null \
                   || echo "unknown date")
        printf "  [%2d] %s\t\t\t(%s)\n" $(( i+1 )) "${names[$i]}" "$ts_human"
    done
    echo "-----------------------------------------------------------------------------"
    echo "  Enter numbers separated by spaces, or press Enter to skip."
    read -rp "  Selection: " selection

    [[ -z "$selection" ]] && { echo "  Skipping WiFi config."; return; }

    # Build wpa_supplicant.conf on the rootfs.
    local wpa_conf="$rootfs/etc/wpa_supplicant/wpa_supplicant.conf"
    sudo mkdir -p "$(dirname "$wpa_conf")"
    sudo tee "$wpa_conf" > /dev/null <<'ENDWPA'
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=PL

ENDWPA

    local copied=0
    for token in $selection; do
        # Validate input is a number in range.
        if ! [[ "$token" =~ ^[0-9]+$ ]] || (( token < 1 || token > ${#names[@]} )); then
            echo "  WARNING: '$token' is not a valid selection — skipped."
            continue
        fi
        local idx=$(( token - 1 ))
        local ssid="${names[$idx]}"

        # Extract PSK via nmcli (requires root; the script already runs as root).
        local psk
        psk=$(nmcli -s -t -f 802-11-wireless-security.psk \
                connection show "$ssid" 2>/dev/null \
              | cut -d: -f2-)

        if [[ -z "$psk" ]]; then
            echo "  WARNING: Could not read PSK for '$ssid' — skipped."
            echo "           (Network may use EAP/enterprise auth, or lacks a stored secret.)"
            continue
        fi

        step "Adding WiFi network: $ssid"
        sudo tee -a "$wpa_conf" > /dev/null <<ENDNET
network={
    ssid="$ssid"
    psk="$psk"
}
ENDNET
        (( copied++ )) || true
    done

    if (( copied > 0 )); then
        sudo chmod 600 "$wpa_conf"
        # Enable wpa_supplicant for wlan0 via systemd symlink.
        local systemd_wpa="$rootfs/etc/systemd/system/multi-user.target.wants"
        sudo mkdir -p "$systemd_wpa"
        local svc_src="/lib/systemd/system/wpa_supplicant@.service"
        local svc_dst="$systemd_wpa/wpa_supplicant@wlan0.service"
        [[ -f "$rootfs$svc_src" ]] && \
            sudo ln -sf "$svc_src" "$svc_dst" 2>/dev/null || true
        step "$copied WiFi network(s) written to wpa_supplicant.conf."
    else
        echo "  No valid networks selected — skipping WiFi config."
    fi
}

# configure_px4() {
#     local rootfs="$1"
#     local startup_script="$rootfs/home/bdd/px4_startup.sh"

#     if [[ ! -f "$startup_script" ]]; then
#         echo "  WARNING: $startup_script not found — skipping PX4 autostart setup."
#         return
#     fi
#     if [[ ! -x "$startup_script" ]]; then
#         echo "  WARNING: $startup_script is not executable — skipping PX4 autostart setup."
#         return
#     fi

#     tee "${rootfs}/home/bdd/start_px4_if_gpio.sh" > /dev/null <<'EOF'
# #!/usr/bin/env bash
# set -euxo pipefail

# exec >> /home/bdd/px4.log 2>&1
# echo "=== PX4 start wrapper $(date) ==="

# if [ "$(gpioget gpiochip0 25)" = "1" ]; then
#     echo "GPIO 25 is high, launching PX4"
#     cd /home/bdd/TMP/PX4-Autopilot
#     exec /usr/bin/taskset -c 2 ./build/scumaker_pilotpi_arm64/bin/px4 -d -s posix-configs/rpi/pilotpi_mc.config
# else
#     echo "GPIO 25 is low, not launching PX4"
#     exit 0
# fi
# EOF

#     chmod +x "${rootfs}/home/bdd/start_px4_if_gpio.sh"
#     # chown bdd:bdd "${rootfs}/home/bdd/start_px4_if_gpio.sh"

#     local svc_file="$rootfs/etc/systemd/system/px4_autostart.service"
#     sudo tee "$svc_file" > /dev/null <<'EOF'
# [Unit]
# Description=PX4 Autostart on PilotPi
# After=network.target systemd-udev-settle.service
# Wants=systemd-udev-settle.service

# [Service]
# Type=simple
# User=bdd
# Group=bdd
# WorkingDirectory=/home/bdd/TMP/PX4-Autopilot
# ExecStartPre=/bin/sleep 5
# ExecStart=/home/bdd/start_px4_if_gpio.sh
# Restart=on-failure
# RestartSec=3

# [Install]
# WantedBy=multi-user.target
# EOF

#     local wants_dir="$rootfs/etc/systemd/system/multi-user.target.wants"
#     sudo mkdir -p "$wants_dir"
#     sudo ln -sf /etc/systemd/system/px4_autostart.service \
#         "$wants_dir/px4_autostart.service"

#     rm -rf "${rootfs}/home/bdd/px4.log"

#     step "PX4 autostart via systemd service."
# }


# -- Cleanup (registered as EXIT trap) ----------------------------------------
cleanup() {
    sync 2>/dev/null || true
    [[ -n "$BOOTFS" ]] && sudo umount "$BOOTFS" 2>/dev/null || true
    [[ -n "$ROOTFS" ]] && sudo umount "$ROOTFS" 2>/dev/null || true
    [[ -n "$BOOTFS" ]] && rmdir "$BOOTFS" 2>/dev/null || true
    [[ -n "$ROOTFS" ]] && rmdir "$ROOTFS" 2>/dev/null || true
}


# -- Argument parsing ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)   IMAGE="$2";              shift 2 ;;
        --ip)      STATIC_IP="$2";          shift 2 ;;
        --gateway) GATEWAY="$2";            shift 2 ;;
        --no-perf) PERF_OPTS=false;         shift   ;;
        --force)   FORCE=true;              shift   ;;
        --only=*)  ONLY="${1#--only=}";     shift   ;;
        -h|--help) usage ;;
        -*)        echo "Unknown option: $1"; usage ;;
        *)         DEVICE="$1"; shift ;;
    esac
done

# Validate --only value.
if [[ -n "$ONLY" ]]; then
    case "$ONLY" in
        flash_sd|configure_boot|configure_user|configure_ssh|configure_network|configure_wifi|configure_px4) ;;
        *) die "--only: unknown function '$ONLY'. Valid values: flash_sd configure_boot configure_user configure_ssh configure_network configure_wifi configure_px4" ;;
    esac
fi

[[ -z "$DEVICE" ]]   && { echo "Error: device not specified."; usage; }
[[ ! -b "$DEVICE" ]] && die "'$DEVICE' is not a block device."

# -- Main ----------------------------------------------------------------------
BOOTFS=$(mktemp -d /tmp/rpi-bootfs-XXXXXX)
ROOTFS=$(mktemp -d /tmp/rpi-rootfs-XXXXXX)
trap cleanup EXIT

if [[ -n "$ONLY" ]]; then
    case "$ONLY" in
        flash_sd)
            flash_sd "$DEVICE" "$IMAGE"
            ;;
        configure_boot)
            mount_partitions "$DEVICE"
            configure_boot "$BOOTFS" "$PERF_OPTS"
            ;;
        configure_user)
            mount_partitions "$DEVICE"
            configure_user "$BOOTFS" "$ROOTFS"
            ;;
        configure_ssh)
            mount_partitions "$DEVICE"
            configure_user "$BOOTFS" "$ROOTFS"
            ;;
        configure_network)
            mount_partitions "$DEVICE"
            configure_network "$ROOTFS" "$STATIC_IP" "$GATEWAY"
            ;;
        configure_wifi)
            mount_partitions "$DEVICE"
            configure_wifi "$ROOTFS"
            ;;
        configure_px4)
            mount_partitions "$DEVICE"
            configure_px4 "$ROOTFS"
            ;;
    esac
else
    flash_sd          "$DEVICE" "$IMAGE"
    configure_boot    "$BOOTFS" "$PERF_OPTS"
    configure_user    "$BOOTFS" "$ROOTFS"
    configure_ssh     "$BOOTFS" "$ROOTFS"
    configure_network "$ROOTFS" "$STATIC_IP" "$GATEWAY"
    configure_wifi    "$ROOTFS"
    configure_px4     "$ROOTFS"
fi

step "Syncing and unmounting..."
sync
cleanup

trap - EXIT
BOOTFS=""
ROOTFS=""

echo ""
if [[ -n "$ONLY" ]]; then
    echo "Done: ran '$ONLY' on $DEVICE."
    case "$ONLY" in
        flash_sd)
            echo "  Flashed  : ${IMAGE:-<no image, mount only>}"
            ;;
        configure_boot)
            echo "  cmdline.txt + config.txt written (perf opts: ${PERF_OPTS})"
            ;;
        configure_user)
            echo "  userconf.txt written (user: bdd / password: 1111)"
            ;;
        configure_ssh)
            echo "  SSH enabled, authorized_keys installed"
            ;;
        configure_network)
            echo "  Static IP : ${STATIC_IP}   Gateway: ${GATEWAY}"
            ;;
        configure_wifi)
            echo "  wpa_supplicant.conf updated"
            ;;
        configure_px4)
            echo "  px4_autostart.service installed and enabled"
            ;;
    esac
else
    echo "SD card ready."
    echo "  Device   : $DEVICE"
    [[ -n "$IMAGE" ]] && echo "  Flashed  : $IMAGE"
    echo "  User     : bdd       Password : 1111"
    echo "  Static IP: ${STATIC_IP}   Gateway  : ${GATEWAY}"
    echo "  Perf opts: ${PERF_OPTS}"
    echo ""
    echo "Insert SD into Raspberry Pi and boot."
    echo "Discover Pi on network : sudo nmap -sP 192.168.*.*"
    echo "SSH with password      : ssh bdd@<IP>"
    echo "SSH with key           : ssh bdd@<IP>  (key already installed)"
fi
