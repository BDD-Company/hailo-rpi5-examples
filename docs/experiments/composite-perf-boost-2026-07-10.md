# Composite performance report — two rig-config sessions, 2026-07-10

**What this is:** the combined, end-to-end performance effect on `bdd@bdd-sd9-mandarin` of *all*
changes made in two Claude Code sessions on 2026-07-10, measured against the state of the box before
either session ran.

| | Session | What it did |
|---|---|---|
| S1 | `63ef38a1-fda3-4097-8c9a-52f5072b8785` | Killed the SC16IS752 IRQ storm; removed `isolcpus`. Ran the measurement matrix. |
| S2 | `c45333a3-369d-48fd-b340-8695b511c4e8` | Audited `config.txt`/`cmdline.txt` vs vanilla bookworm; removed dead settings; fixed a latent SPI0 bug; tested and rejected `mitigations=off`. |

Baseline = the box as it was **before 2026-07-10 12:48 UTC** (pre-S1): recording on, tiling ladder
live, the `sc16is752` interrupt storm running underneath, and the whole 51-thread app squeezed onto 2
of 4 cores by `isolcpus`.

---

## TL;DR — combined boost, production configuration

Recording **on**, tiling ladder **live** — the shape the drone actually flies.

| metric | baseline (pre-S1) | after both sessions | change |
|---|---|---|---|
| **e2e p50** (capture→command) | 215.2 ms | **55.1 ms** | **−74% · 3.9× faster** |
| **e2e p95** | 317.8 ms | **75.1 ms** | −76% |
| **e2e p99** | 424.5 ms | **81.9 ms** | **−81% · 5.2× faster** |
| **e2e max** | 790.0 ms | **109.3 ms** | **−86% · 7.2× faster** |
| **frames ≤ 100 ms** | 0.6% | **100%** | +99.4 pts |
| **throughput** | 21.6 fps | **30.0 fps** | **+39%** (now camera-capped) |
| idle CPU (cpu0, box at rest) | 42.8% | **0.1%** | storm gone |
| die temp @ low load (mean) | 57.3 °C | **51.6 °C** | −5.7 °C |
| thermal throttling | `throttled=0x0` | `throttled=0x0` | never throttled, 2400 MHz pinned |

The 50–120 ms capture→command goal is now met **at p99**, with recording and tiling both enabled — a
configuration that previously could not hold 200 ms even at the median.

> The p50 baseline is stated as the measured run B (recording + static tiles + storm + 2 cores,
> 215.2 ms). The true pre-S1 production shape also had the tiling ladder flapping on top, so its real
> p50 was **≥ 215 ms** — this table is if anything conservative about the size of the win.

---

## What actually moved the needle — the S1 matrix

S1 ran a 7-cell matrix (150 s each, cooled ≤58 °C before each run, `throttled=0x0` throughout, zero
stalls / zero tracebacks) that isolates each change. `--tiles 1x1` (static whole-frame) in A–F so the
tiling ladder couldn't confound; G is the real production shape.

| run | config | p50 | p95 | p99 | max | ≤100 ms | fps |
|---|---|---|---|---|---|---|---|
| **A** | no-rec · storm · 2 cores | 63.0 | 139.6 | 174.1 | 279.6 | 80.8% | 28.1 |
| **B** | **rec** · storm · 2 cores | **215.2** | 317.8 | 424.5 | 790.0 | 0.6% | 21.6 |
| **C** | no-rec · storm **pinned** · 2 cores | 39.1 | 40.7 | 42.0 | 47.3 | 100% | 30.0 |
| **D** | **rec** · storm **pinned** · 2 cores | 65.6 | 82.1 | 91.6 | 137.4 | 99.6% | 30.0 |
| **E** | no-rec · **no storm · 4 cores** | 38.7 | 39.9 | 40.6 | 43.2 | 100% | 30.0 |
| **F** | **rec** · **no storm · 4 cores** | 53.0 | 70.7 | 78.1 | 114.0 | 100% | 30.0 |
| **G** | **production** (rec + tiling ladder) | 55.1 | 75.1 | 81.9 | 109.3 | 100% | 30.0 |

**B is the baseline. G is the delivered state.** Everything between is the decomposition.

### Contribution of each change (recording path, B → F → G)

| step | change | source | p50 | p99 | what it bought |
|---|---|---|---|---|---|
| B → D | disable `sc16is752-spi1` IRQ storm | **S1** | 215.2 → 65.6 | 424.5 → 91.6 | **the whole win.** Tail stops existing: the 104k-IRQ/s FIFO-50 handler stopped preempting app threads 50,855×/s. |
| D → F | remove `isolcpus` (2 → 4 cores) | **S1** | 65.6 → 53.0 | 91.6 → 78.1 | ~12 ms at p50 on the recording path. The app wants 227% of a core when recording; 2 cores minus an 87%-of-a-core storm left ~1.05 — that starvation is why B managed only 21.6 fps. |
| F → G | tiling ladder live (production shape) | (feature) | 53.0 → 55.1 | 78.1 → 81.9 | ~neutral. The runtime tiling ladder rides for free (53 branch switches, 0 stalls). |
| — | config hygiene + `mitigations` decision | **S2** | ≈ 0 | ≈ 0 | no latency delta by design — see below. |

**Reading of the matrix:** ~90% of the latency recovery is the interrupt-storm fix alone; the
`isolcpus` removal is a smaller but real second win that only shows up once recording competes for CPU.
The long-standing "recording costs 3×, so `--no-record` is mandatory for any latency claim" rule is
retired: recording now costs **1.37×** (53.0/38.7), not 3.42× — because it was never recording, it was
the storm.

---

## What session S2 added (beyond latency)

S2's config audit produced **zero measured latency change** — that was the correct and intended
outcome, confirmed across 6 benched runs matching the healthy baseline (p50 54–55 ms, 100% ≤100 ms,
30 fps). Its value is elsewhere:

- **Fixed a real correctness bug.** `dtoverlay=uart3` had been hand-added on the box and muxed
  GPIO8/9 as TXD3/RXD3 — *those are SPI0's CE0/MISO pins* — silently killing SPI0 (`dtparam=spi=on`
  produced no `spi0` master and no `/dev/spidev0.*`). Removed.
- **Removed dead configuration** at no cost: `uart2`, `uart3`, `dtparam=spi=on` (dead as configured),
  `i2c_arm`/`i2c_vc`, `gpu_mem=512` (inert on Pi 5 — VideoCore VII uses CMA), `elevator=deadline`
  (no-op since Linux 5.0), and a duplicate `quiet`.
- **Evaluated and rejected `mitigations=off`.** Real ~9% win on a `getppid()` syscall microbench, but
  **statistically indistinguishable at app e2e** (the app is camera-bound, StageA ≈ 25 ms, not
  syscall-bound). Kept the safe default ON — the box is Tailscale-reachable and runs a flight
  controller. (Methodology note for the record: one ON run fluked to p50 118 ms from a 0.86 s
  `capture_request` startup stall; it reproduced in neither arm. Replicate ≥3× before trusting an
  app-latency delta on this rig.)

So S2 **locked in** S1's gain (cleaned config now authoritative, `setup_sd.sh` removed so a
`deploy.sh` can't clobber it) and paid down correctness/hygiene debt without moving the latency number.

---

## Net attribution

- **Session 63ef38a1 (S1)** delivered essentially 100% of the measured performance boost — the two
  boot-config changes (`sc16is752` off, `isolcpus` gone) account for the entire 3.9×/5.2×/7.2×
  latency improvement and the 21.6 → 30 fps throughput recovery.
- **Session c45333a3 (S2)** added no latency but fixed a latent SPI0 fault, stripped dead config, and
  made the box's cleaned boot state authoritative.
- **Combined, versus the pre-S1 baseline:** capture→command p50 **215 → 55 ms**, p99 **425 → 82 ms**,
  max **790 → 109 ms**, ≤100 ms budget **0.6% → 100%**, throughput **21.6 → 30 fps**, all at a lower
  die temperature with no thermal throttling.

---

## Caveats & methodology

- All runs on `bdd-sd9-mandarin` (Pi 5, kernel 6.12.62, Hailo-8 on PCIe Gen3, one IMX477 noir cam),
  no cooling by design, on-rig bouncing-ball scene, constant lighting (auto-exposure railed at
  9992 µs / gain 16.0 in every evening run — a clean control across S2's runs).
- Metric is **capture→command (sensor→command) end-to-end latency**, the project's primary optimization
  target — not raw FPS. FPS is now camera-capped at 30, so the throughput gain understates the real
  headroom the storm was stealing.
- Baseline p50 is run B (215.2 ms). No "recording + tiling ladder + storm + 2 cores" run was taken —
  it was pointless once B couldn't hold 200 ms — so the production baseline was if anything worse.
- `throttled=0x0` and 2400 MHz clock held in every run; none of these deltas are thermal artifacts.

## Sources

- Handoffs written by these sessions: `BDD/experiments/rpi-boot-fixes-handoff.md` (S1),
  `BDD/experiments/boot-config-audit-2026-07-10.md` and `BDD/experiments/SD_CARD_HANDOFF.md` (S2).
- Session transcripts: `63ef38a1-fda3-4097-8c9a-52f5072b8785`, `c45333a3-369d-48fd-b340-8695b511c4e8`.
