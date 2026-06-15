#!/usr/bin/env python3
"""durable_tee.py — a tee(1) that survives an abrupt power cut.

Reads stdin, writes everything to BOTH the terminal (stdout, for the live tmux
pane) and a log file, and fsync()s the log file so an unclean power-off loses at
most a fraction of a second instead of the kernel's whole writeback window.

bdd.sh pipes the app's combined stdout+stderr through this, so the SINGLE
_DEBUG/BDD_<ts>.log holds everything — Python logging AND native GStreamer /
libcamera / interpreter-traceback output — durably, with no second file.

Durability policy (mirrors the validated helpers.DurableFileHandler):
  * fsync immediately when a chunk contains ERROR / CRITICAL / FATAL — the line
    explaining a crash must reach the card before the power goes,
  * otherwise fsync at most once per FSYNC_INTERVAL_S (default 0.5 s), so a burst
    of output is not throttled by an fsync per write,
  * a final fsync on EOF / signal.
The containing directory is fsync'd once up front so the new file itself cannot
vanish as a 0-byte husk.

We read the RAW fd with os.read() (not a buffered readline): select() reflects
the fd's true readability, and every available byte is written + fsync-considered
immediately — a buffered reader would slurp a burst into Python's own buffer
where select() can't see it, stranding the last lines (e.g. an ERROR right before
a crash) unwritten until more data happens to arrive.

Usage:  some_command 2>&1 | durable_tee.py /path/to/file.log
"""
import sys, os, select, time, signal

FSYNC_INTERVAL_S = 0.5
URGENT = (b"ERROR", b"CRITICAL", b"FATAL")


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: durable_tee.py <logfile>\n")
        return 2
    path = sys.argv[1]

    # Unbuffered binary append: each write() is a syscall straight to the page
    # cache (no stdio buffer to lose), and fsync() then pushes it onto the card.
    logf = open(path, "ab", buffering=0)
    # Make the directory entry durable once, so the file can't be a husk.
    dfd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)

    log_fd = logf.fileno()
    out_fd = sys.stdout.fileno()
    in_fd = sys.stdin.fileno()
    last_fsync = 0.0
    dirty = False

    # Flush + fsync on SIGTERM/SIGHUP (tmux kill-session) so the tail survives a
    # clean-ish shutdown too. Default SIGINT handling (KeyboardInterrupt) is fine.
    def _final(*_):
        try:
            os.fsync(log_fd)
        except OSError:
            pass
        os._exit(0)
    signal.signal(signal.SIGTERM, _final)
    signal.signal(signal.SIGHUP, _final)

    try:
        while True:
            r, _, _ = select.select([in_fd], [], [], FSYNC_INTERVAL_S)
            now = time.monotonic()
            if r:
                chunk = os.read(in_fd, 65536)
                if not chunk:          # EOF — the app exited
                    break
                logf.write(chunk)      # unbuffered -> straight to the kernel
                dirty = True
                try:                   # the pane may be gone; never let that stop logging
                    os.write(out_fd, chunk)
                except OSError:
                    pass
                if any(tag in chunk for tag in URGENT):
                    os.fsync(log_fd)
                    last_fsync, dirty = now, False
            if dirty and (now - last_fsync) >= FSYNC_INTERVAL_S:
                os.fsync(log_fd)
                last_fsync, dirty = now, False
    finally:
        try:
            os.fsync(log_fd)
        except OSError:
            pass
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
