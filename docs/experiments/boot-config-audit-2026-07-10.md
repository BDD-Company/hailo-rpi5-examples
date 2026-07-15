# bdd-sd9 boot-config audit — 2026-07-10

> Ref: Claude Code session `c45333a3-369d-48fd-b340-8695b511c4e8`.

Reviewed `bdd@bdd-sd9-mandarin` `/boot/firmware/{config.txt,cmdline.txt}` against vanilla
`2025-11-24-raspios-bookworm-arm64.img`, explained every meaningful difference, removed dead
settings, and measured the one open performance lever (`mitigations=off`). All changes are live on
the box; originals backed up at `/home/bdd/_bootcfg_backup/*.orig` (and on the laptop).

Hardware: Raspberry Pi 5, kernel 6.12.62, Hailo-8 on PCIe, one IMX477 (noir) camera, external
Auterion PX4 FMU on USB `/dev/ttyACM0`. No cooling by design.

---

## 1. Every meaningful line vs vanilla bookworm

### KEEP — deliberate and correct

| Setting | What it does | Verdict |
|---|---|---|
| `force_turbo=1` | Pins ARM at 2400 MHz, disables DVFS | **Keep** — kills frequency-scaling latency spikes; core to the latency work. Sets the OTP warranty bit (already set). |
| `over_voltage_delta=50000` | +50 mV for stability at pinned clock | **Keep** — pairs with `force_turbo`. |
| `dtparam=pciex1_gen=3` | Hailo-8 M.2 PCIe link at Gen3 (2× Gen2 BW) | **Keep** — this is the inference data path. |
| `camera_auto_detect=0` + `dtoverlay=imx477,cam0/cam1` | Explicit, deterministic camera bring-up | **Keep**. |
| `enable_uart=1`, `dtoverlay=uart0`, `dtoverlay=disable-bt` | GPIO14/15 primary UART for RC input; frees the PL011 from BT | **Keep** (confirmed). This is why cmdline drops vanilla's `console=serial0,115200`. |
| `vc4-kms-v3d`, `max_framebuffers=2`, `display_auto_detect=1` | GL/KMS stack | **Keep** — the `--preview` path over wayvnc/labwc uses it. |
| `cfg80211.ieee80211_regdom=PL` (cmdline) | Wi-Fi regulatory domain = Poland | **Keep** — harmless, correct region. |

### REMOVED — dead weight or actively harmful (all proven zero app cost)

| Removed | Why it was safe / why it mattered |
|---|---|
| `isolcpus` (cmdline) | Reservation was dead; squeezed the whole app onto 2 of 4 cores. Already handled earlier today — see memory. |
| `dtoverlay=sc16is752-spi1` | Unused SPI-UART bridge that caused a 104k-IRQ/s storm eating ~87% of a core. Already handled earlier today. |
| **`dtoverlay=uart3`** | Muxed GPIO8/9 as TXD3/RXD3 — **those are SPI0's CE0/MISO pins**, so it silently killed SPI0. No consumer. Hand-added on the box (not in `setup_sd.sh`). |
| `dtoverlay=uart2` | GPIO4/5, no consumer. Hand-added on the box. |
| `dtparam=spi=on` | Produced **no `spi0` master / no `/dev/spidev0.*`** at all, because uart3 owned its pins. Dead as configured. |
| `dtparam=i2c_arm=…`, `dtparam=i2c_vc=on` | Unused. Cameras use their own dedicated CSI I2C buses (i2c-4/-6/-10/-11), not these. |
| `gpu_mem=512` | **Inert on Pi 5** — VideoCore VII uses CMA, not the legacy split. `vcgencmd get_mem gpu` = 4M regardless; all 8 GB stayed with ARM; CMA unchanged at 64 MB. |
| `elevator=deadline` (cmdline) | Deprecated no-op on modern blk-mq; mmcblk already defaults to `mq-deadline`. |
| duplicate `quiet` (cmdline) | Cosmetic. |

### Cleaned files now live

`cmdline.txt`:
```
console=tty1 root=PARTUUID=a82fe8f7-02 rootfstype=ext4 fsck.repair=yes rootwait quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=PL
```
`config.txt` functional lines: `audio=on`, `enable_uart/uart0/disable-bt`, `camera_auto_detect=0` +
`imx477,cam0/cam1`, `vc4-kms-v3d`/`max_framebuffers=2`/`display_auto_detect=1`/`disable_fw_kms_setup=1`,
`arm_64bit`/`disable_overscan`/`arm_boost`, `pciex1_gen=3`, `force_turbo=1`, `over_voltage_delta=50000`,
`[cm4] otg_mode=1`, `[cm5] dwc2,dr_mode=host`.

---

## 2. Things that look alarming but are BENIGN

- **126 kernel `WARNING … find_vma … hailo_vdma_buffer_map` backtraces.** All fire inside a 0.44 s
  window at pipeline startup — the out-of-tree `hailo_pci` driver calls `find_vma` without holding
  `mmap_lock`. Not per-frame, not config-related, nothing to fix from boot config.
- **Firmware-injected cmdline params** (`numa=fake=8`, `cgroup_disable=memory`, `pci=pcie_bus_safe`,
  `coherent_pool=1M`, `system_heap.max_order=0`, …) are Pi 5 bootloader defaults, identical on
  vanilla. Not introduced by this setup.
- **Only one camera enumerates** despite `imx477,cam1`. Expected — the zoom sensor is temporarily off
  and the app's `cameras:` list has it commented out. Kept per your call; costs only a failed I2C
  probe at boot.

---

## 3. Performance / latency measurements

Benchmarks: `--DEBUG`, default config (recording + tiling ladder on), on-rig bouncing-ball scene.
Lighting was constant across all evening runs (auto-exposure railed at `9992 µs / gain 16.0` in every
run), so the camera regime was identical — a clean control.

### `mitigations=off` — measured, then reverted

**Microbench (light-independent, cpu0-pinned, best-of-3):**

| Test | mitigations ON | mitigations OFF | delta |
|---|---|---|---|
| 4M `getppid()` syscall storm | 1.145 s | 1.040 s | **~9% faster** |
| 200k pipe ctx-switch ping-pong | 1.280 s | 1.273 s | ~0.5% (noise) |

So `mitigations=off` (disables BHB clearing; CSV2 is hardware and stays) is a real ~9% win on raw
syscall cost.

**App e2e latency (3 runs each, identical scene):**

| | p50 | p99 | max | ≤100 ms |
|---|---|---|---|---|
| OFF ×3 | 53.1 / 55.2 / 54.3 | 80.7 / 84.5 / 78.8 | 97 / 104 / 107 | 100 / 100 / 100% |
| ON ×3 | 54.0 / 57.5 / 54.3 | 78.8 / 93.2 / 85.6 | 88 / 122 / 107 | 100 / 99.5 / 99.9% |

**Statistically indistinguishable.** The ~9% syscall saving doesn't reach e2e because the app is
camera-bound (StageA ≈ 25 ms), not syscall-bound.

**Decision: keep the safe default (ON).** No app benefit, and the box is Tailscale-reachable and runs
a flight controller — not worth the hardening tradeoff.

> Trap for the record: the very first ON run measured p50 **118 ms** and looked like a big regression.
> It was a transient fluke — that single run had an 0.86 s `capture_request` startup stall that
> cascaded into buffer drops. It reproduced in neither the ON nor the OFF replication. Always replicate
> ≥3× before trusting an app-latency delta on this rig.

### Net latency effect of the whole audit

The cleaned config matches the historical healthy baseline (p50 ≈ 54–55 ms, 100% ≤100 ms, 30 fps,
`throttled=0x0`, 0 tracebacks) across six benched runs. The big latency wins (dropping `isolcpus` and
the sc16is752 storm) were already banked earlier today; this pass **removed dead configuration without
costing any latency** and **fixed a real correctness bug** (uart3 silently disabling SPI0).

---

## 4. Provisioning status (updated 2026-07-10)

`scripts/setup_sd.sh` — the old boot-config provisioner — has been **removed** from the repo. No
remaining repo script touches `/boot/firmware` (`deploy.sh` only rsyncs `BDD/`, `scripts/`,
`requirements.txt` into tmpfs). So:

- The cleaned `/boot/firmware/{config.txt,cmdline.txt}` on the box is now **authoritative** — a normal
  `deploy.sh` run will not clobber it, and there's no stale script left to reintroduce the removed cruft.
- Trade-off: there's also no longer a scripted way to reproduce this boot config on a fresh flash. If
  the SD is ever reimaged from stock, reapply by hand from `/home/bdd/_bootcfg_backup/*.orig` (the
  cleaned versions) or from the laptop copies in the audit scratchpad. Consider keeping a copy of the
  two cleaned files in the repo (e.g. `scripts/boot/`) as documentation-of-record.
