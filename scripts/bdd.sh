#! /usr/bin/env bash

readonly SESSION_NAME="bdd-app"

kill_resource_holder() {
    local resource="$1"
    for p in $(fuser -v ${resource} 2>&1 | tail -n-1 | awk '{print $3}');
    do
        echo ${p} holds ${resource} killing it...
        kill -9 ${p} ||:
    done
}

if [[ -z "$TMUX" ]]; then
    # --- Bootstrap: not yet inside tmux ---
    if ! command -v tmux >/dev/null; then
        echo "tmux is required. Install with: sudo apt install -y tmux" >&2
        exit 1
    fi

    # Kill any prior session; tmux sends SIGHUP to its children, which should
    # take down the previous python app and release /dev/hailo0, /dev/video*.
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null ||:
    sleep 1

    # Defense in depth: if anything didn't die cleanly, force-free the devices
    # before the new run starts.
    kill_resource_holder "/dev/hailo0"
    kill_resource_holder "/dev/video*"

    # Re-exec self inside a fresh detached tmux session, then attach.
    quoted=$(printf '%q ' "$0" "$@")
    tmux new-session -d -s "$SESSION_NAME" "$quoted"

    if [[ -t 1 ]]; then
        exec tmux attach -t "$SESSION_NAME"
    else
        echo "Started detached in tmux session '$SESSION_NAME'."
        echo "Attach: tmux attach -t $SESSION_NAME"
        exit 0
    fi
fi

# --- Inner: running inside tmux ---
cd ./hailo-rpi5-examples/
. ./setup_env.sh

readonly start_time=$(date +%Y%m%d-%H%M%S)
mkdir -p ./_DEBUG/ ||:

# Share this run id with the app so the log file and the MKV segments all carry
# the same timestamp.
export BDD_START_TIME="${start_time}"

# Mirror EVERYTHING (Python logging + native GStreamer/libcamera stderr) into a
# SINGLE log file, still shown live in the tmux pane. durable_tee.py fsyncs that
# file (immediately on ERROR/CRITICAL/FATAL, else ~2x/s) so an abrupt power cut
# loses at most a fraction of a second -- see scripts/setup_durability.sh for the
# kernel-side writeback tuning that backs this up.
exec > >(python3 ./scripts/durable_tee.py "./_DEBUG/BDD_${start_time}.log") 2>&1

set -ex

if git status ;
then
    git -P log -n1 --oneline
    git -P describe --long --always --dirty ||:
    git -P diff ||:
fi

#    export G_MESSAGES_DEBUG=all
#    export GST_TRACERS="latency;stats"
#   export GST_DEBUG='*:INFO'
export GST_DEBUG=3
export LIBCAMERA_LOG_LEVELS=1

# Display for --preview. This Pi runs a Wayland compositor (labwc) mirrored over the
# network by wayvnc, so point GStreamer's waylandsink at the compositor socket. Without
# these, autovideosink falls back to kmssink (direct DRM scanout) which wayvnc can't
# capture -> the remote preview looks like a slideshow. DISPLAY=:0 kept as an X11
# fallback for a locally-attached monitor / non-Wayland host.
export DISPLAY=:0
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"

python \
    ./BDD/app.py \
    -i rpi \
    "$@"
