#!/usr/bin/env bash
#
# setup_durability.sh — make BDD logs & videos survive an abrupt power cut,
#                       software-only (no UPS / supercap / hardware changes).
#
# WHAT IT DOES (the two *persistent, system-level* changes):
#
#   1. /etc/sysctl.d/90-drone-durability.conf
#        Shrinks the kernel page-cache writeback window from the stock 5 s/30 s
#        down to ~1 s, so dirty data is pushed toward the card within ~1 s even
#        for files nothing explicitly fsyncs (the in-progress video chunk and the
#        tee'd console log). This is what bounds VIDEO loss to ~1 s — the recorder
#        does not fsync segments; it relies on this writeback window.
#
#   2. /etc/fstab  ->  add `commit=1` to the ext4 root filesystem
#        Forces the ext4 journal to commit data+metadata every 1 s instead of the
#        default 5 s, tightening the worst-case loss window and the 0-byte-husk gap.
#
# The launch script additionally pipes the app's combined stdout/stderr through
# scripts/durable_tee.py, which fsyncs the single log file on ERROR/CRITICAL/FATAL
# immediately (and ~2x/s otherwise), so the crash *cause* is never lost; that lives
# in the repo and deploys with the normal code rsync.
#
# Measured on bdd-sd9 (sysrq-b = simulated power loss): with this writeback tuning,
# completed video segments and logs survive an unclean shutdown to within ~1 s;
# without any of this, recent logs/videos are lost as 0-byte husks.
#
# USAGE:
#   On the Pi itself:          sudo ./setup_durability.sh
#   Against a mounted SD root: sudo ./setup_durability.sh --root /mnt/sdcard-root
#   Preview only (no writes):  ./setup_durability.sh --dry-run [--root DIR]
#   Skip the live remount/sysctl (just edit files): sudo ./setup_durability.sh --no-apply
#
set -euo pipefail

ROOT=""
DRYRUN=0
APPLY=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)     ROOT="${2%/}"; shift 2 ;;
    --dry-run)  DRYRUN=1; shift ;;
    --no-apply) APPLY=0; shift ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# When targeting an alternate (mounted) root we are NOT on that system, so never
# try to apply settings to the *running* kernel.
[[ -n "$ROOT" ]] && APPLY=0

SYSCTL_FILE="$ROOT/etc/sysctl.d/90-drone-durability.conf"
FSTAB="$ROOT/etc/fstab"

read -r -d '' SYSCTL_CONTENT <<'EOF' || true
# Power-loss durability for the BDD drone (installed by setup_durability.sh).
# Push dirty page-cache data toward storage within ~1 s instead of the stock 5 s/30 s,
# so an abrupt power cut loses at most ~1 s of not-yet-fsync'd data.
vm.dirty_writeback_centisecs = 100
vm.dirty_expire_centisecs    = 100
# Absolute (RAM-size-independent) thresholds so writeback starts early and the
# amount of in-flight dirty data stays small.
vm.dirty_background_bytes    = 16777216
vm.dirty_bytes               = 67108864
EOF

say() { printf '\033[1;36m[durability]\033[0m %s\n' "$*"; }

require_writable() {
  # Root is only needed to write a root-owned location (real /etc). When targeting a
  # user-writable mounted SD or fake-root, no sudo is required.
  [[ $DRYRUN -eq 1 ]] && return 0
  local dir="$1"
  mkdir -p "$dir" 2>/dev/null || true
  if [[ ! -w "$dir" ]]; then
    echo "cannot write to $dir — re-run with sudo (or fix the mount permissions)" >&2; exit 1
  fi
}

# ---------------------------------------------------------------------------
# 1. sysctl writeback tuning
# ---------------------------------------------------------------------------
say "sysctl file -> $SYSCTL_FILE"
if [[ $DRYRUN -eq 1 ]]; then
  echo "----- would write -----"; printf '%s\n' "$SYSCTL_CONTENT"; echo "-----------------------"
else
  require_writable "$(dirname "$SYSCTL_FILE")"
  printf '%s\n' "$SYSCTL_CONTENT" > "$SYSCTL_FILE"
  say "wrote $SYSCTL_FILE"
fi

# ---------------------------------------------------------------------------
# 2. fstab: add commit=1 to the ext4 root line (idempotent)
# ---------------------------------------------------------------------------
say "fstab     -> $FSTAB (ensure commit=1 on ext4 / )"
[[ -f "$FSTAB" ]] || { echo "no fstab at $FSTAB" >&2; exit 1; }

NEW_FSTAB="$(awk '
  $1 !~ /^#/ && $2=="/" && $3=="ext4" {
    n=split($4,a,","); opts="";
    for(i=1;i<=n;i++){ if(a[i] !~ /^commit=/){ opts=(opts==""?a[i]:opts","a[i]) } }
    opts=opts",commit=1";
    printf "%s\t%s\t%s\t%s\t%s\t%s\n", $1,$2,$3,opts,$5,$6;
    changed=1; next
  }
  { print }
  END { if(!changed) exit 3 }
' "$FSTAB")" || {
  st=$?
  if [[ $st -eq 3 ]]; then
    echo "ERROR: no 'ext4  /' line found in $FSTAB — refusing to guess." >&2; exit 1
  fi
  echo "awk failed ($st)" >&2; exit 1
}

if diff -q <(printf '%s\n' "$NEW_FSTAB") "$FSTAB" >/dev/null 2>&1; then
  say "fstab already has commit=1 on root — no change"
elif [[ $DRYRUN -eq 1 ]]; then
  echo "----- fstab diff (proposed) -----"
  diff -u "$FSTAB" <(printf '%s\n' "$NEW_FSTAB") || true
  echo "---------------------------------"
else
  require_writable "$(dirname "$FSTAB")"
  cp -a "$FSTAB" "${FSTAB}.bak.$(date +%Y%m%d-%H%M%S)"
  printf '%s\n' "$NEW_FSTAB" > "$FSTAB"
  say "updated $FSTAB (backup kept alongside)"
  echo "----- new root line -----"; grep -E '[[:space:]]/[[:space:]].*ext4' "$FSTAB" || true; echo "-------------------------"
fi

# ---------------------------------------------------------------------------
# 3. Apply to the running system (on-box only)
# ---------------------------------------------------------------------------
if [[ $APPLY -eq 1 && $DRYRUN -eq 0 ]]; then
  if [[ $EUID -ne 0 ]]; then echo "live apply needs root; re-run with sudo or use --no-apply" >&2; exit 1; fi
  say "applying sysctl live..."
  sysctl -p "$SYSCTL_FILE"
  say "remounting / with commit=1 live..."
  mount -o "remount,commit=1" /
  say "verify: $(findmnt -no OPTIONS /)"
elif [[ $APPLY -eq 0 && $DRYRUN -eq 0 ]]; then
  say "files written; NOT applied live (reboot or 'sysctl --system' + remount to activate)."
fi

say "done. (Durable Python logging deploys with the BDD code; video durability relies on the writeback tuning above.)"
