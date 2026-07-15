# Assignment: remove two boot-time settings that cripple the Pi 5 rig

**For:** the coding agent in the project that owns SD-card provisioning.
**Status:** applied by hand to the live box `bdd-sd9` on 2026-07-10, verified by measurement.
Provisioning must now be updated so a re-image cannot reintroduce them.

---

## Summary

| # | File | Action |
|---|------|--------|
| 1 | `config.txt` | Comment out `dtoverlay=sc16is752-spi1` |
| 2 | `cmdline.txt` | Delete the `isolcpus=` token |

Together these took the drone app's capture→command latency from **p50 > 200 ms to p50 55.1 ms**
and throughput from **21.6 to 30 fps**, in production configuration.

---

## Change 1 — `config.txt`: disable the SC16IS752 overlay

Find where the provisioning script writes the `config.txt` overlay block. It currently emits:

```ini
# Enable sc16is752 over SPI1
dtoverlay=sc16is752-spi1
```

Replace with (comment out the `dtoverlay`, keep and extend the explanatory comment):

```ini
# Enable sc16is752 over SPI1
# Disabled 2026-07-10: the chip's level-triggered IRQ line sits permanently
# asserted, so the sc16is7xx threaded handler re-reads it over SPI forever —
# ~104,000 interrupts/sec, burning ~87% of a CPU core at SCHED_FIFO prio 50,
# preempting every application thread. Nothing uses /dev/ttySC0 or /dev/ttySC1.
# Re-enable ONLY if the two extra UARTs are actually wired up AND the IRQ
# storm is fixed (pull-up / trigger polarity on the IRQ GPIO).
# dtoverlay=sc16is752-spi1
```

### Why

The SC16IS752 is an SPI-to-dual-UART bridge. Its interrupt pin is stuck asserted, so the kernel's
threaded handler spins forever servicing it. Measured on the idle box, before the fix:

- IRQ on the SPI controller fired **104,444 times per second** — 1.38 billion in a 3h41m uptime,
  a flat constant rate, not a transient.
- The handler kthread `irq/169-spi1.0` runs at **`SCHED_FIFO` priority 50**, so it preempts every
  thread the application owns (all `SCHED_OTHER`).
- Total cost measured in isolation on an otherwise-empty core: **86.8% of that core.** Note `ps`
  shows only ~36% — it counts the threaded handler but not the hardirq time, understating by ~2.4×.
- Consequence: the app's threads were being involuntarily preempted **25,350 times per second**.

Nothing on the box has ever opened `/dev/ttySC0` or `/dev/ttySC1`, and no code references them.

---

## Change 2 — `cmdline.txt`: remove `isolcpus`

The script writes a `cmdline.txt` line containing `isolcpus=2`. (The live box had been hand-edited
at some point to `isolcpus=2,3`.) **Delete the `isolcpus=` token entirely.** Everything else on the
line stays exactly as it is.

```diff
- console=tty1 root=PARTUUID=%s rootfstype=ext4 fsck.repair=yes rootwait elevator=deadline isolcpus=2 quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=PL quiet
+ console=tty1 root=PARTUUID=%s rootfstype=ext4 fsck.repair=yes rootwait elevator=deadline quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=PL quiet
```

### Why

The reservation existed to pair with a `taskset -c 2` that pinned an **on-board** PX4 build. That
build is no longer used — the rig flies an **external** Auterion PX4 FMU v6X over USB
(`/dev/ttyACM0`). Nothing ever ran on the isolated cores: `cpu2` and `cpu3` logged **zero** non-idle
jiffies since boot, while the entire 51-thread application was squeezed onto `cpu0` and `cpu1`.

With `isolcpus` gone, the kernel's scheduler spreads the load evenly across all four cores with no
pinning whatsoever (measured: 61.1 / 60.9 / 61.6 / 62.2 percent under load).

---

## Do NOT change

- `dtparam=spi=on` — backs `spidev`, unrelated to the SC16IS752.
- `enable_uart=1` and the `uart0` / `uart2` / `uart3` overlays.
- Both `imx477` camera overlays (`cam0`, `cam1`).
- `force_turbo=1`, `over_voltage_delta=50000`, `dtparam=pciex1_gen=3`.
- Anything touching the USB FMU on `/dev/ttyACM0`.

---

## Hard constraints

- `cmdline.txt` must remain **exactly one line**, LF-terminated, with **no carriage returns**. The
  Pi firmware silently fails to parse a multi-line file, producing an unbootable card.
- Keep the `root=PARTUUID=` templating intact. That value is per-card and must still be read from
  the flashed partition, never hardcoded.
- Preserve `rootfstype=ext4`, `rootwait`, and `fsck.repair=yes`.

---

## Verification after flashing and first boot

```bash
cat /sys/devices/system/cpu/isolated        # expect: empty
nproc; nproc --all                          # expect: 4 and 4   (was 2 and 4)
ls /dev/ttySC*                              # expect: no such file
ps -eo comm | grep '^irq/'                  # expect: no irq/NNN-spi1.0 thread
grep -i '1f00054000.spi' /proc/interrupts   # expect: no match
```

Then confirm the box idles near zero — sample `/proc/stat` for ~8 s with nothing running; every core
should read under ~1% busy. **Before the fix, `cpu0` idled at 42.8%.**

> Do not grep for a fixed IRQ number. The number is not stable across kernel configs — match on the
> device name (`1f00054000.spi`) instead.

---

## Expected impact

Measured on the drone app in production configuration (recording on, tiling ladder live), 150 s,
4493 samples, `--tiles` untouched:

| metric | before | after |
|---|---|---|
| e2e p50 | > 200 ms | **55.1 ms** |
| e2e p99 | 424.5 ms | **81.9 ms** |
| e2e max | 790.0 ms | **109.3 ms** |
| frames ≤ 100 ms | 0.6% | **100%** |
| throughput | 21.6 fps | **30.0 fps** |
| idle CPU (cpu0) | 42.8% | **0.1%** |

The die also runs *cooler* at low load (51.6 °C vs 57.3 °C mean), because the interrupt storm was
itself a heat source. Nothing throttled in any run; the clock stayed pinned at 2400 MHz.

---

## Rollback

Keep copies of both files before writing. **If the box fails to boot it needs physical access to the
SD card — there is no remote recovery.** The single most likely cause of an unbootable card is a
stray newline or CR in `cmdline.txt`; verify with `od -c`, not by eye.

---

## Reference

`scripts/setup_sd.sh` in the `hailo-rpi5-examples` repo is a **stale, unused** copy of this
provisioning logic, but it shows the exact shape of the code to look for:

- line 173 — the `printf` that writes `cmdline.txt` (contains `isolcpus=2`)
- line 187 — `dtoverlay=sc16is752-spi1`

Do not edit that file; it provisions nothing. It is useful only for locating the analogous lines in
the real project.

## Optional cleanup (low priority)

`elevator=deadline` is also present in `cmdline.txt` and has been a no-op since Linux 5.0 removed the
parameter; the box runs 6.12.62, so the kernel ignores it. Harmless, but dead weight. To actually set
an I/O scheduler, use a udev rule or `/sys/block/*/queue/scheduler`.
