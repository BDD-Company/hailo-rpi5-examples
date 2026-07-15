# SD-card / boot-config handoff — bake these into the provisioner

**For:** whoever owns the SD-card image / provisioning project for the BDD drone box
(`bdd-sd9-mandarin`, a Raspberry Pi 5).
**From:** boot-config audit on the live box, 2026-07-10. Every "PROVEN" claim below is backed by
on-rig measurements (summarized at the end).
**Ref:** Claude Code session `c45333a3-369d-48fd-b340-8695b511c4e8`.
**Why this exists:** the old in-repo provisioner (`hailo-rpi5-examples/scripts/setup_sd.sh`) was
deleted, and the real provisioning lives in your project. The fixes below were applied **by hand** to
the running box. Unless you bake them into the image, the next reflash silently undoes them.

Target hardware: **Raspberry Pi 5**, kernel 6.12.x (bookworm), **Hailo-8** accelerator on the M.2/PCIe
slot, **one IMX477** (noir) camera on `cam0`, external Auterion **PX4 FMU over USB** (`/dev/ttyACM0`).
The box runs with **no active cooling by design** (thermal throttling is an accepted test condition).

---

## 1. Drop-in files (copy-paste ready)

On bookworm these live at **`/boot/firmware/config.txt`** and **`/boot/firmware/cmdline.txt`**
(older images: `/boot/config.txt`).

### `/boot/firmware/config.txt`

```ini
# For more options and information see http://rptl.io/configtxt

# Enable audio (loads snd_bcm2835)
dtparam=audio=on

# Enable full UART0 (GPIO14/15) for RC input; disable Bluetooth to free the PL011
enable_uart=1
dtoverlay=uart0
dtoverlay=disable-bt

# -- Cameras ------------------------------------------------------------------
# Explicit, deterministic bring-up (no firmware auto-detect).
# NOTE: this box uses IMX477, NOT ov5647. cam1 is present for a returning 2nd camera.
camera_auto_detect=0
dtoverlay=imx477,cam0
dtoverlay=imx477,cam1

# -- Display / framebuffer -----------------------------------------------------
# Headless box, but the --preview path (wayvnc/labwc) uses the GL/KMS stack.
display_auto_detect=1
auto_initramfs=1
dtoverlay=vc4-kms-v3d
max_framebuffers=2
disable_fw_kms_setup=1

# -- CPU / memory --------------------------------------------------------------
arm_64bit=1
disable_overscan=1
arm_boost=1

# -- PCIe ----------------------------------------------------------------------
# Run PCIe at Gen 3 full speed — this is the Hailo-8 inference data path.
dtparam=pciex1_gen=3

# -- Performance ---------------------------------------------------------------
# Pin the ARM clock at max: removes frequency-scaling latency spikes.
force_turbo=1
# Small voltage bump for stability at the pinned clock.
over_voltage_delta=50000

[cm4]
otg_mode=1

[cm5]
dtoverlay=dwc2,dr_mode=host
```

### `/boot/firmware/cmdline.txt` (single line)

```
console=tty1 root=PARTUUID=<ROOT_PARTUUID> rootfstype=ext4 fsck.repair=yes rootwait quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=PL
```

> **`<ROOT_PARTUUID>` is per-card** — substitute the real root partition PARTUUID during provisioning
> (the live box currently has `a82fe8f7-02`; do not hardcode that). Everything else is verbatim.

---

## 2. What changed vs a stock Pi OS config, and why

### KEEP — deliberate, proven or clearly correct

| Setting | Purpose | Status |
|---|---|---|
| `force_turbo=1` + `over_voltage_delta=50000` | Pin ARM at 2400 MHz; kill DVFS latency spikes | **KEEP.** Sets the OTP warranty bit (already set on this unit). |
| `dtparam=pciex1_gen=3` | Hailo-8 PCIe link at Gen3 (2× Gen2 bandwidth) | **KEEP.** Officially "unsupported" on Pi 5 but stable here; it's the inference data path. |
| `camera_auto_detect=0` + `dtoverlay=imx477,cam0/cam1` | Deterministic camera bring-up | **KEEP.** Use **imx477**, not ov5647 (the old script had `ov5647,cam0` — wrong for this box). |
| `enable_uart=1`, `dtoverlay=uart0`, `dtoverlay=disable-bt` | GPIO14/15 primary UART for RC input | **KEEP.** This is why cmdline omits `console=serial0,115200`. |
| `vc4-kms-v3d`, `max_framebuffers=2`, `display_auto_detect=1` | GL/KMS stack | **KEEP** — the `--preview` over VNC uses it. |
| `cfg80211.ieee80211_regdom=PL` (cmdline) | Wi-Fi regulatory domain | **KEEP** (set to your region). |

### REMOVE — dead weight or actively harmful (do NOT re-add)

| Removed | Why |
|---|---|
| `dtoverlay=sc16is752-spi1` | **Biggest real win.** Unused SPI-UART bridge stuck in a ~104k IRQ/s storm; its SCHED_FIFO-50 handler + hardirq burned **~87% of a CPU core** and preempted every app thread. |
| `isolcpus=2` (or `isolcpus=2,3`) in cmdline | Dead reservation (was for an on-board PX4 that no longer exists). Stranded 2 of 4 cores — the whole app ran on cores 0/1 only. |
| `dtoverlay=uart3` | **Correctness bug.** Muxes GPIO8/9 as TXD3/RXD3 — those are **SPI0's CE0/MISO pins**, so it silently disables SPI0. No consumer. |
| `dtoverlay=uart2` | GPIO4/5, no consumer. |
| `dtparam=spi=on` | Produced **no `spi0` / no `/dev/spidev0.*`** because uart3 owned its pins. Dead. (Re-add only if you actually wire an SPI device — and then drop uart3.) |
| `dtparam=i2c_arm=…`, `dtparam=i2c_vc=on` | Unused. Cameras use their own dedicated CSI I2C buses. |
| `gpu_mem=512` | **Inert on Pi 5** (VideoCore VII uses CMA, not the legacy split; firmware reports gpu=4M and leaves all RAM to ARM). |
| `elevator=deadline` (cmdline) | Deprecated no-op on blk-mq; mmc already defaults to `mq-deadline`. |
| duplicate `quiet` (cmdline) | Cosmetic. |

### MEASURED AND REJECTED — do NOT add

| Candidate | Result |
|---|---|
| `mitigations=off` | Real ~9% on a raw syscall microbench, but **zero measurable app-latency benefit** (app is camera-bound, not syscall-bound). Not worth the security trade-off on a network-reachable box that flies a drone. **Leave mitigations at default.** |

---

## 3. Evidence behind the "PROVEN" claims

All on-rig, Pi 5, kernel 6.12.62, HEAD of the app, bouncing-ball scene, `--DEBUG`. e2e = camera
capture → control command sent. Budget target: ≤100 ms.

- **sc16is752 storm removed + isolcpus removed** (the two big wins, measured before/after reboot):
  - `--no-record`: e2e p50 **63 → 39 ms**, p99 **174 → 42 ms**, max **280 → 47 ms**.
  - recording on: e2e p50 **215 → 65 ms**, p99 **425 → 92 ms**; DET fps **21.6 → 30**.
  - Recording's cost fell from **3.4× to ~1.4×** e2e; box idle CPU **42.8% → 0.1%**.
  - Production config (recording + tiling on): **p50 55 / p99 82 / max 109 ms, 100% ≤100 ms, 30 fps,
    `throttled=0x0`**.
- **This audit's cleanup** (drop uart2/3, dead spi, i2c, gpu_mem): 6 benched runs on the cleaned
  config, all healthy — p50 53–57 ms, ~100% ≤100 ms, 0 tracebacks, cameras enumerate, `throttled=0x0`.
  Latency-neutral (as expected — these were dead settings) but **fixes the SPI0 breakage** and cuts a
  failed boot-time probe.
- **mitigations=off**: microbench getppid storm 1.145 → 1.040 s (~9%); app e2e ON p50 54/58/54 vs OFF
  53/55/54 — indistinguishable across 3 runs each.

---

## 4. Verify after applying to a fresh card

Boot, then check:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E 'isolcpus|mitigations'   # expect: no output
cat /sys/devices/system/cpu/isolated                                # expect: empty
nproc                                                               # expect: 4
ls /dev/ttySC*                                                      # expect: No such file (storm gone)
vcgencmd get_throttled                                              # expect: throttled=0x0
vcgencmd measure_clock arm                                          # expect: ~2400000000 (force_turbo)
rpicam-hello --list-cameras                                         # expect: imx477 enumerated
grep -c hailo_vdma_buffer_map <(dmesg)                              # a burst at boot is BENIGN (see note)
```

Idle CPU should sit near ~0% (a busy idle = the IRQ storm came back → sc16is752 overlay slipped in).

---

## 5. Caveats / notes

- **Per-card PARTUUID** in cmdline (see §1) — must be substituted, not copied.
- **PCIe Gen3** is unofficial on Pi 5; stable on this unit but validate on each board.
- **`force_turbo=1` + no cooling** is intentional — the box is meant to run hot and throttle under
  field heat. Do not "fix" thermals; do keep an eye on `vcgencmd get_throttled` during validation.
- **Second camera**: `imx477,cam1` is kept for a returning zoom camera even though only `cam0` is
  populated now. Harmless — costs one failed I2C probe at boot. Parameterize sensor type/count if your
  provisioner supports it.
- **Benign at boot**: a burst of `WARNING … find_vma … hailo_vdma_buffer_map` kernel backtraces fires
  once during pipeline startup (Hailo's out-of-tree driver). Not per-frame, not fixable from boot
  config — ignore.
- **App files** (`BDD/`, `scripts/`, `models/`) are deployed separately into tmpfs via the app repo's
  `scripts/deploy.sh` (rsync) and are wiped on reboot **by design** — that is NOT part of this boot
  config and should stay out of it.
