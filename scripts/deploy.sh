#!/usr/bin/env bash
#
# Deploy project files to a Raspberry Pi over SSH/SCP.
#
# Security model: target directories on the RPi are mounted as tmpfs,
# so all deployed data is wiped automatically on power loss.
# This script:
#   1. Ensures the tmpfs mounts exist on the RPi (creates them if needed)
#   2. Copies project files into those tmpfs-backed directories
#
# Usage:
#   ./scripts/deploy.sh [RPI_HOST]
#
# RPI_HOST defaults to "bdd-rpi-drone-lan" (from ~/.ssh/config).
# You can also pass an IP/hostname directly:
#   ./scripts/deploy.sh bdd@192.168.1.42

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Configuration -----------------------------------------------------------

RPI_HOST="${1:-bdd-rpi-drone-lan}"

# Base directory on the RPi where the project lives
REMOTE_BASE="/home/bdd/hailo-rpi5-examples"

# Directories to mount as tmpfs on the RPi (relative to REMOTE_BASE).
# Everything in these dirs is volatile — wiped on power loss.
TMPFS_DIRS=(
    "BDD"
    "scripts"
    "models"
)

# tmpfs size limit per mount (adjust as needed)
TMPFS_SIZE="64M"

# Files/directories to deploy (relative to PROJECT_ROOT -> remote relative to REMOTE_BASE)
DEPLOY_ITEMS=(
    "BDD"
    "scripts"
    "requirements.txt"
)

# Excluded patterns (rsync-style)
EXCLUDES=(
    "__pycache__"
    "*.pyc"
    ".git"
)

# --- Functions ----------------------------------------------------------------

log()  { printf '\033[1;32m>>>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

check_ssh() {
    log "Checking SSH connectivity to ${RPI_HOST} ..."
    if ! ssh -o ConnectTimeout=5 "${RPI_HOST}" true 2>/dev/null; then
        die "Cannot reach ${RPI_HOST} over SSH. Check your connection / SSH config."
    fi
}

setup_tmpfs_mounts() {
    log "Ensuring tmpfs mounts on ${RPI_HOST} ..."

    for dir in "${TMPFS_DIRS[@]}"; do
        remote_path="${REMOTE_BASE}/${dir}"

        # Check if already a tmpfs mount
        ssh "${RPI_HOST}" bash <<REMOTE_EOF
set -euo pipefail

# Create the directory if it doesn't exist
sudo mkdir -p "${remote_path}"
sudo chown bdd:bdd "${remote_path}"

# If not already a tmpfs mount, mount it
if ! mountpoint -q "${remote_path}" 2>/dev/null; then
    echo "Mounting tmpfs at ${remote_path} ..."
    sudo mount -t tmpfs -o size=${TMPFS_SIZE},mode=0755,uid=\$(id -u bdd),gid=\$(id -g bdd) tmpfs "${remote_path}"
else
    echo "${remote_path} is already a tmpfs mount."
fi
REMOTE_EOF
    done
}

deploy_files() {
    log "Deploying files to ${RPI_HOST}:${REMOTE_BASE} ..."

    local exclude_args=()
    for pattern in "${EXCLUDES[@]}"; do
        exclude_args+=(--exclude "${pattern}")
    done

    for item in "${DEPLOY_ITEMS[@]}"; do
        local src="${PROJECT_ROOT}/${item}"
        if [[ ! -e "${src}" ]]; then
            warn "Source '${src}' does not exist, skipping."
            continue
        fi

        if [[ -d "${src}" ]]; then
            # rsync directory contents (trailing slash = contents of dir)
            rsync -avz --delete \
                "${exclude_args[@]}" \
                "${src}/" \
                "${RPI_HOST}:${REMOTE_BASE}/${item}/"
        else
            # single file
            rsync -avz \
                "${src}" \
                "${RPI_HOST}:${REMOTE_BASE}/${item}"
        fi
    done
}

verify_deployment() {
    log "Verifying deployment ..."
    ssh "${RPI_HOST}" bash <<REMOTE_EOF
set -euo pipefail
echo "--- tmpfs mounts ---"
mount | grep "${REMOTE_BASE}" || echo "(no tmpfs mounts found under ${REMOTE_BASE})"
echo ""
echo "--- deployed files ---"
for dir in ${TMPFS_DIRS[*]}; do
    echo "${REMOTE_BASE}/\${dir}:"
    ls -la "${REMOTE_BASE}/\${dir}" 2>/dev/null | head -5
    count=\$(find "${REMOTE_BASE}/\${dir}" -type f 2>/dev/null | wc -l)
    echo "  (\${count} files total)"
    echo ""
done
REMOTE_EOF
}

# --- Main ---------------------------------------------------------------------

main() {
    log "Deploying to ${RPI_HOST}"
    log "Project root: ${PROJECT_ROOT}"

    check_ssh
    setup_tmpfs_mounts
    deploy_files
    verify_deployment

    log "Deployment complete. Data on RPi is volatile — will be wiped on power loss."
}

main
