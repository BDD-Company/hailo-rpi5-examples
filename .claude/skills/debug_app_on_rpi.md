# How to launch bdd python app on raspberry pi and analyze logs

Raspberry pi is usually connected to the same local network and thus accessible via `bdd-sd9.local` hostname. From this developer machine one can log into the rpi box with user `bdd`, using the ssh key already loaded into the ssh-agent.

## Safety precheck

The BDD app connects to PX4 over USB and **arms the motors** as part of its startup sequence (see `startup_sequence` in `BDD/drone.py`). Before launching, confirm the drone is in a safe state — props off or otherwise secured on a bench.

## Steps

1. **SSH connectivity** — `ssh bdd@bdd-sd9.local` (expect `bdd-sd9` hostname). If this fails, nothing below will work.

2. **Copy the model file referenced by `scripts/bdd.sh`** to the rpi. The HEF name in `bdd.sh`'s `HAILO_MODEL` is the source of truth and is updated occasionally; don't hard-code an old name. Resolve and copy in one go:
   ```
   HEF=$(awk -F\" '/^export HAILO_MODEL=/{print $2}' /media/Pets/BDD/hailo-rpi5-examples/scripts/bdd.sh | xargs -n1 basename)
   scp "/media/Pets/BDD/_MODELS/$HEF" bdd@bdd-sd9.local:/home/bdd/models/
   ```
   Failure mode if you skip this: HailoRT logs `HAILO_OPEN_FILE_FAILURE` during pipeline startup and the Hailo plugin segfaults during state transition — the app dies in <1 s, and on libcamerasrc paths it can leave mavsdk telemetry callbacks piling up for minutes as the slow async shutdown drains.

3. **Verify `HAILO_MODEL` in `scripts/bdd.sh`** points at the model just copied:
   ```
   ssh bdd@bdd-sd9.local "grep ^export.HAILO_MODEL ~/hailo-rpi5-examples/scripts/bdd.sh && ls -la ~/models/\$(awk -F\\\" '/^export HAILO_MODEL=/{print \$2}' ~/hailo-rpi5-examples/scripts/bdd.sh | xargs -n1 basename)"
   ```

4. **Sync sources to the rpi** — this step is **mandatory before every launch**, not optional. `~/bdd.sh` on the remote is a symlink to `~/hailo-rpi5-examples/scripts/bdd.sh`; if `scripts/` hasn't been synced (or was synced before that subdir existed in the repo), the symlink dangles and `./bdd.sh` will fail with no useful message.
   - EITHER rsync `/media/Pets/BDD/hailo-rpi5-examples/BDD/` and `scripts/` into `bdd@bdd-sd9.local:/home/bdd/hailo-rpi5-examples/`. A safe one-liner that excludes per-run artifacts (`_DEBUG/` archived logs, caches, mp4s) so a local archive in `_DEBUG/_runs/` doesn't get shipped *back* to the Pi:
     ```
     rsync -v --mkpath -rc \
       --exclude='*venv*' --exclude=doc --exclude='*__pycache__*' \
       --exclude='**/.*' --exclude='*test_data/**' --exclude=community_projects/ \
       --exclude='_DEBUG/' --exclude='*/**.mp4' \
       --include='*.py' --include='*.sh' --chmod=+x \
       /media/Pets/BDD/hailo-rpi5-examples bdd@bdd-sd9.local:
     ```
   - OR execute the `sync to remote` vscode task with `bdd-sd9.local` as the only parameter (the prompt asks for an IP, but a hostname works too).
   - Sanity-check after sync: `ssh bdd@bdd-sd9.local 'readlink -f ~/bdd.sh && test -x ~/bdd.sh && echo OK'` — must print `OK`.

5. **Launch** — ssh into the rpi and execute `./bdd.sh --DEBUG` from user `bdd`'s home directory.
   - `bdd.sh` re-execs itself inside a detached tmux session named `bdd-app`.
   - Running it non-interactively over ssh (`ssh -T bdd@host './bdd.sh --DEBUG' < /dev/null`) is fine — bdd.sh detects the missing TTY and prints `Started detached in tmux session 'bdd-app'.` then exits, while the app continues inside tmux.

6. **Locate the log** — a new file `BDD_<YYYYMMDD-HHMMSS>.log` appears in `/home/bdd/hailo-rpi5-examples/_DEBUG/` within a few seconds of launch.
   ```
   ssh bdd@bdd-sd9.local 'ls -t ~/hailo-rpi5-examples/_DEBUG/BDD_*.log | head -1'
   ```

7. **Monitor and verify the app is healthy.** Drive monitoring as a single fire-and-collect with a **hard time cap** — never condition-poll without a deadline (the app can spin in an error loop without crashing, and a condition that waits for "100 detections OR tmux gone OR Traceback" will hang forever when none of those happen). The remote `sleep` runs on the Pi; the ssh process exits cleanly when it returns, so this is the right shape for the agent's `Bash run_in_background=true`:
   ```
   ssh bdd@bdd-sd9.local '
     sleep 60
     L=$(ls -t ~/hailo-rpi5-examples/_DEBUG/BDD_*.log | head -1)
     echo "log=$L  tmux=$(tmux ls 2>/dev/null || echo none)"
     echo "lines=$(wc -l < $L)  GOT_DETECTIONS=$(grep -c "GOT DETECTIONS" $L)  dups=$(grep -c "Skipped duplicated" $L)"
     echo "overflow=$(grep -c "callback queue overflown" $L)  tracebacks=$(grep -c Traceback $L)  segfaults=$(grep -c "Segmentation fault\|core dumped" $L)"
     grep "picamera_thread alive" $L | tail -3
     grep "!!! LATENCY" $L | tail -3
     true   # keep exit 0 even if a downstream grep had zero matches
   '
   ```
   The trailing `true` is important: `grep` returns 1 when there are no matches, which would otherwise propagate as a "failed" exit code through the shell `&&` chain or the background-task status and falsely report the run as broken when it was actually healthy and your monitoring just looked for something correctly absent (e.g., "Using libcamerasrc" on a default-backend run).

   Signs of a healthy run:
   - Many `!!! GOT DETECTIONS, objects detected: N` lines from `drone_controller.py` (~15–20 fps expected).
   - `picamera_thread alive: pushed N frames so far ... per-frame avg: capture_wait=… loop_proc=…` lines from `app_base.py` (instrumentation; appears every 100 frames).
   - `!!! LATENCY sensor→command(e2e): … = queue_wait … + processing …` lines from `drone_controller.py` (emitted when a control command is actually sent).
   - Telemetry lines showing `flight_mode: '7 (OFFBOARD)'` and a `landed_state`.
   - `Offboard.set_attitude_rate(...)` or `Offboard.set_position_ned(...)` calls from `drone.py:move_to_target_*`.
   - Tmux session still present: `ssh bdd@bdd-sd9.local 'tmux ls'` should show `bdd-app`.

   Noise to ignore (these are NOT failures):
   - `fatal: not a git repository` early in the log — bdd.sh runs `git status` but only sources are rsync'd, no `.git`.
   - `<ERROR>`-level lines from `app.py` `main()` 344-348 — the deliberate banner announcing DEBUG mode.
   - GStreamer `FIXME ... appsrc ... queue is full` — pipeline-internal, not the app's concern.
   - libcamera/vadisplay `WARN` lines during pipeline init.

   **Real failure signatures** — any of these means the run is broken even if `tmux ls` is still showing `bdd-app`:
   - `Traceback`, `OffboardError`, `Failed to arm`, the tmux session disappearing, or **no `GOT DETECTIONS` lines after 10+ seconds**.
   - **Error loop**: log line count exploding (tens of thousands of lines in <60s) with a single repeated message dominating. The canonical one is `User callback queue overflown` from mavsdk `system.py:87` — it means the drone-controller asyncio loop is stuck and telemetry callbacks are piling up. The app may have already errored and be hanging in shutdown. Count it explicitly (`grep -c "callback queue overflown"`); a healthy run has 0–~5.
   - `gst-stream-error-quark: Internal data stream error` from `app_base.py:bus_call` — the GStreamer pipeline died (e.g., negotiation race like the appsrc `format=time` issue fixed in `pipelines.py`).
   - `HAILO_OPEN_FILE_FAILURE` + later `Segmentation fault (core dumped)` — the HEF in `HAILO_MODEL` doesn't exist on the Pi (step 2 was skipped or stale).

   **ANSI-color gotcha when grepping log levels:** the log writes levels wrapped in ANSI color codes — `<ERROR>` is actually `<\x1b[1;31mERROR\x1b[1;0m>` in the file, so `grep "<ERROR>"` misses every error line. Either grep just the word (`grep "ERROR"`), or strip first: `sed 's/\x1b\[[0-9;]*m//g'`. Same applies to `<WARNING>`, `<INFO>`, `<DEBUG>`.

8. **Terminate**: either `Ctrl-C` inside the attached tmux pane, or from another shell:
   ```
   ssh bdd@bdd-sd9.local 'tmux kill-session -t bdd-app'
   ```
   Verify with `tmux ls` — should report no sessions.

9. **Pull the log down for analysis** if needed:
   ```
   scp bdd@bdd-sd9.local:/home/bdd/hailo-rpi5-examples/_DEBUG/BDD_<timestamp>.log /tmp/
   ```
