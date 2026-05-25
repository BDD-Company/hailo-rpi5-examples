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
        echo "App running detached in tmux session '$SESSION_NAME'."
        echo "Streaming its output below; lines stay in THIS terminal's scrollback."
        echo "Ctrl-C stops the APP (forwarded into tmux). Reattach: tmux attach -t $SESSION_NAME"
        echo

        # Ctrl-C here sends a real SIGINT to the app in the pane (graceful, just
        # like pressing Ctrl-C while attached) instead of only killing the tail.
        trap 'tmux send-keys -t "$SESSION_NAME" C-c 2>/dev/null' INT

        # Foreground the log in the real terminal (no tmux alt-screen) so output
        # accumulates in the client's native scrollback and survives a Pi power-out.
        # The subshell ignores INT so the tail keeps printing the app's shutdown
        # messages; -F waits for latest.log and re-follows when a new run rotates it.
        ( trap '' INT; exec tail -n +1 -F "./hailo-rpi5-examples/_DEBUG/latest.log" ) &
        tail_pid=$!

        # Block until the app's session ends (normal exit, or after Ctrl-C), then
        # stop the tail and drop back to the shell.
        while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
            sleep 1
        done
        sleep 1                                   # let tail flush the last lines
        kill "$tail_pid" 2>/dev/null ||:
        exit 0
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

# Stable name the foreground `tail -F` (see bootstrap) can follow across runs.
ln -sfn "BDD_${start_time}.log" "./_DEBUG/latest.log"

# Mirror stdout/stderr to log file while still showing live in the tmux pane.
exec > >(tee "./_DEBUG/BDD_${start_time}.log") 2>&1

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

export DISPLAY=:0
export HAILO_MODEL="/home/bdd/models/2026-04-22_11n_sh_v5.hef"

python -u \
    ./BDD/app.py \
    --hef-path "${HAILO_MODEL}" \
    -i rpi \
    "$@"

