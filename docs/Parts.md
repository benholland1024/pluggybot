 

# Parts List

This document provides specs + model info for parts used in this project. 

> **Sourcing note:** Pololu (US) parts are stocked by German/EU distributors — mainly [Eckstein-shop.de](https://eckstein-shop.de/Pololu_EN), plus BerryBase, EXP-Tech, Welectron, Botland and TME.eu — so no US import is needed. All prices below are **approximate, incl. 19% VAT, as of July 2026** — re-check before ordering.

---

## Drive system

### Motors — runner-up: 30:1 (not selected)

**Pololu 30:1 Metal Gearmotor 37Dx68L mm 12V with 64 CPR Encoder (Helical Pinion)** — Pololu #4752

- Source: [Eckstein-shop.de](https://eckstein-shop.de/Pololu-301-Metal-Gearmotor-37Dx68L-mm-12V-with-64CPR-EncoderHelical-Pinion-EN) — **€84,43 each** (in stock, 7–9 working days). Datasheet: [pololu.com/product/4752](https://www.pololu.com/product/4752)

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| Gear ratio | 30:1 | — |
| No-load speed @ 12 V | 330 rpm = **34.6 rad/s** | actuator `ctrlrange` / velocity limit (matches sim ~35 rad/s ✓) |
| Stall torque @ 12 V | 14 kg·cm = **1.37 N·m** | actuator `forcerange` (sim uses 1.4 — close enough, or update to 1.37) |
| Stall current @ 12 V | 5.5 A | motor driver sizing (not sim) |
| No-load current | 0.2 A | — |
| Encoder | 64 CPR motor shaft = **1920 CPR at output** (≈0.19°/count) | future odometry / sensor noise model |
| Mass | **200 g** | motor `geom` mass (2× = 400 g of the ~1.1 kg budget) |
| Output shaft | 16 mm long, **6 mm D-shaft** | wheel/hub compatibility (see below) |
| Dimensions | ⌀37 × 68 mm (excl. shaft) | motor geom size |

Matches the original sim numbers exactly, but passed over in favor of the 50:1's docking push force — speed is a low priority for this robot.

### Motors (2×) — ✅ CHOSEN: 50:1 (July 2026)

**Pololu 50:1 Metal Gearmotor 37Dx70L mm 12V with 64 CPR Encoder (Helical Pinion)** — Pololu #4753

- Source: [Eckstein-shop.de](https://eckstein-shop.de/Pololu-501-Metal-Gearmotor-37Dx70L-mm-12V-with-64CPR-EncoderHelical-Pinion-EN) — **€84,43 each**. Datasheet: [pololu.com/product/4753](https://www.pololu.com/product/4753)
- 50:1, **200 rpm** no-load (= 20.9 rad/s), **21 kg·cm = 2.06 N·m** stall, 5.5 A stall, 3200 CPR at output, 205 g, same 6 mm D-shaft.
- ✅ **Applied to sim (July 2026):** `ctrlrange` ±21 rad/s, `forcerange` ±2.06 N·m, wheel-joint `armature="0.012"` (reflected rotor inertia ∝ gear-ratio², up from 0.005 at 30:1) and `damping="0.05"` (the 50:1 gearbox's ~30–35 % torque losses, per Pololu's ~65 % efficiency figure). The damping term also turned out to be load-bearing for sim stability — see SimNotes.md.

### Wheels (2×)

⚠ **Finding:** Pololu's 60–70 mm wheels only fit 3 mm shafts. For the 37D's **6 mm D-shaft**, the verified Pololu path is a 90 mm (or 80 mm) wheel + universal mounting hub.

| Part | Source | Price | Notes |
|---|---|---|---|
| **Pololu Wheel 90×10 mm pair** (#1435–1439) | [Eckstein-shop.de](https://eckstein-shop.de/Pololu-Wheel-90x10mm-Pair-Red-for-Micro-Metal-Gearmotors-EN) | **€11,13 / pair** | Six M3/#4-40 mounting holes matching Pololu universal hubs. Mass: TBD (verify on datasheet) |
| **Pololu Universal Aluminum Mounting Hub, 6 mm shaft, M3 holes (2-pack)** (#1999) | [Eckstein-shop.de](https://eckstein-shop.de/PololuUniversalAluminumMountingHubfor6mmShaft2CM3Holes2-PackEN) | **€11,95 / 2-pack** | Set-screw hub for the 6 mm D-shaft; wheel bolts to hub |
| Alt: Pololu Multi-Hub Wheel 80×10 mm (2-pack) | [Eckstein-shop.de](https://eckstein-shop.de/Pololu-Multi-Hub-Wheel-w-Inserts-for-3mm-and-4mm-Shafts-8010mm-Black-2-pack-EN) | €14,20 / 2-pack | Inserts are 3/4 mm only — **TBD: verify it accepts the 6 mm universal hub** before buying |

✅ **Decided + applied to sim (July 2026):** 90×10 mm wheels accepted → wheel geom r = 0.045 m, half-width 0.005. With the chosen 50:1 motor, top speed is 20.9 rad/s × 0.045 ≈ **0.94 m/s** and stall push force ≈ 2.06/0.045 ≈ 46 N per wheel — ample. Note the larger radius destabilized the sim's pitch dynamics until tire compliance (`solref="0.05 1"`) and gearbox damping were modeled — see SimNotes.md.

---

## Chassis & mechanical

| Part | Source | Price | Specs → sim |
|---|---|---|---|
| **Pololu Ball Caster with 3/4″ metal ball** (#955) | [EXP-Tech](https://www.exp-tech.de/zubehoer/mechanische-bauteile/5551/pololu-ball-caster-with-3/4-metal-ball) (in stock); also [BerryBase](https://www.berrybase.at/pololu-ball-caster-0-75-zoll-metallkugel-abs-gehaeuse-hoehenverstellbar-fuer-kleine-roboter), [TME.eu](https://www.tme.eu/en/details/pololu-955/accessories-for-robotics-and-rc/pololu/ball-caster-with-3-4-metal-ball/) | **≈ €4,50** (€3,72 net) | Ball ⌀ 19 mm → caster `sphere` geom; height 0.83″–≈1″ (21–25 mm) adjustable via spacers → chassis ground clearance. Mass: TBD (verify on datasheet) |
| Pololu 37D Metal Gearmotor Bracket (pair) | [Eckstein Pololu mounts category](https://eckstein-shop.de/Pololu-Motor-Mounts-Wheel-EN) | TBD | Sets motor axle height above chassis plate |
| Chassis plate | TBD (laser-cut acrylic/alu, or Misumi/igus stock profiles) | TBD | Track width target 0.21 m — set by motor bracket spacing, verify wheel-to-wheel once brackets chosen |

---

## Vision

Two viable routes — see Open decisions.

### ✅ CHOSEN — Option A: 2× Raspberry Pi Camera Module 3 (DIY stereo, July 2026)

- Source: [Welectron](https://www.welectron.com/Official-Raspberry-Pi-Camera-Module-3) — **€25,50 each** (≈ €51 for the pair); also [BerryBase](https://www.berrybase.de/en/raspberry-pi-camera-module-3-12mp). Specs: [raspberrypi.com](https://www.raspberrypi.com/documentation/accessories/camera.html)

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| Sensor / resolution | Sony IMX708, 11.9 MP, 4608 × 2592 | camera resolution in renders |
| Horizontal FOV | 66° | — |
| **Vertical FOV** | **41°** | - |
| Focus | motorized PDAF, ~10 cm–∞ | near-field plug alignment OK |
| Mass | **4 g** each | camera geom mass (negligible) |
| Dimensions | 25 × 24 × 11.5 mm | mount design |

- Mounting at **60 mm baseline**: modules are 25 mm wide, so 60 mm center-to-center leaves ~35 mm clear between them — fine. Needs a rigid printed/laser-cut bracket (stereo calibration dies if the baseline flexes) — no off-the-shelf 60 mm stereo mount; plan a custom bracket. Note: Pi 5 has two CSI ports, so both cameras connect natively, but frame capture is not hardware-synchronized (software sync is usually acceptable at robot speeds ≤1.5 m/s).

### Option B (not selected): Luxonis OAK-D Lite (integrated stereo + depth ASIC)

- Source: [MYBOTSHOP.DE](https://www.mybotshop.de/Luxonis-DepthAI-OAK-D-Lite_1) — **€192,95**; [Reichelt](https://www.reichelt.com/de/en/shop/product/luxonis_depthai_oak-d_lite_fixed-focus-324637) — €199,40 (fixed-focus **backordered until ~17 Aug 2026**); also [Botland](https://botland.store/modules-smart-cameras/20955-oak-d-lite.html). Specs: [docs.luxonis.com](https://docs.luxonis.com/hardware/products/OAK-D%20Lite)

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| **Stereo baseline** | **75 mm** | ⚠ sim assumes 60 mm — **update stereo camera separation to 0.075 m** if chosen |
| Mono (depth) cameras | 2 × 640 × 480 | depth quality model |
| Depth range | MinZ ≈ 35 cm (≈20 cm extended mode), MaxZ ≈ 10 m ± 10 % | ⚠ 35 cm min depth is tight for terminal plug-docking approach |
| Mono camera FOV | TBD (verify on datasheet) | camera `fovy` |
| RGB camera | 12.3 MP, 4K/30 | — |
| Compute | on-board depth + 1.4 TOPS NN (RVC2), USB-C | offloads Pi 5 |
| Mass | TBD (verify on datasheet, ~60 g class) | camera geom mass |

---

---

## Arm & docking (milestone 6)

### Naming

**Mast** = the fixed vertical column. **Lift** = the carriage that travels up it (and
the actuator driving that). **Telescoping arm** = the horizontal extension carrying the
plug. This is Hello Robot Stretch's vocabulary, and the plan already borrows its
architecture: base owns x/yaw, lift owns z, arm owns reach.

### ⚠ The mass budget is the binding constraint — read before choosing anything

Measured in sim (`models/world.xml`, headless force probe), pushing a plug into a wall
socket at outlet height (0.30 m) reacts back on the robot, and the current 1.14 kg
chassis cannot take it:

| Push at 0.30 m | Caster load | Drive-wheel load | Outcome |
|---|---|---|---|
| 0 N (static) | 4.12 N | 7.08 N | — |
| 2 N | 0.76 N | 10.44 N | near tipping |
| 3 N | **0 N** | 11.77 N | **caster lifts — robot pitches back** |

Against insertion forces measured by the Schuko spike: **0.7 N** perfectly aligned,
**6.1 N** at 2 mm lateral, **7.8 N** at 2° yaw. So the robot tips at ~2.4 N — it can
only complete a *well-aligned* insertion.

Three consequences that drive the part choices below (updated after modelling —
the armed 2.34 kg robot was re-measured with ramped loads):

1. ~~A wall-brace foot~~ — **falsified in sim.** A wall contact is one-way: it can only
   push the robot *away*, the same direction as the insertion reaction, so bracing pads
   add to the overturning moment instead of resisting it (with-brace measured worse
   than without). The pads survive as **alignment feelers**: both tips touching the
   wall squares yaw mechanically across a 0.14 m base and references insertion depth.
2. **Battery position is a real design variable, not packaging** — in two axes now.
   x = +0.05 (slightly ahead of centre) for tipping margin; **y = +0.06 as a
   counterweight** for the arm assembly hanging at y = −0.05, without which the robot
   veers 26 cm right over 4 m of open-loop driving.
3. **The docking force budget is ~3 N, and it is the controller's problem.** Armed
   robot, push at outlet height: forward docking holds ~3 N then *slides*; backward
   holds ~4 N then goes caster-light. Spike insertion forces: 0.7 N aligned, 6.1 N at
   2 mm off. So insertion must stay ≲3 N → terminal alignment ≲1 mm (visual servo),
   softer RCC springs, or active-drive tricks. **Forward docking stays the
   recommendation** (mapping and cameras face forward); the failure mode is a benign
   slide, not the destructive tip previously feared.

### Lift and telescoping arm — 2× linear actuator

**igus drylin® E lead-screw stepper linear actuator, NEMA11** — igus DLE-LA-0001

- Source: [igus.com/product/DLE-LA-0001](https://www.igus.com/product/DLE-LA-0001). Price: **TBD** — igus quotes stroke-configured units through their configurator, not a fixed list price. Get a quote for both axes together.

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| Max thrust | **50 N** | actuator `forcerange` — 6× the worst-case 7.8 N insertion, ample |
| Holding torque | 0.12 N·m | holds the lift against gravity unpowered |
| Lead screw | dryspin® high helix DST 6.35 × 5.08 | 5.08 mm travel per revolution |
| Linear feed per step | **0.0254 mm** (1.8° step) | far finer than the ±3 mm docking budget; positioning is not the limit |
| Motor flange | NEMA11 / 28 mm | mount design |
| Lubrication | none required (dryspin®) | — |
| Stroke | configurable — want **~0.25 m lift**, **~0.20 m reach** | joint `range` |
| Mass | TBD (verify on datasheet) | body mass — matters, see mass budget above |

Stroke rationale: the lift must span outlet heights 0.26–0.38 m in `room_1.xml` **and**
carry the docking camera high enough to keep a 0.38 m outlet in frame at close range
(the fixed 0.18 m eye loses it below 0.40 m — measured). Reach is set by parking the
base at ~0.25 m rather than the current 0.6 m standoff; a 0.6 m cantilever on a 1.14 kg
robot is unaffordable.

### Plug

**Rewireable Schuko CEE 7/7 plug (Type F)** — e.g. [Leads Direct rewireable right-angle](https://leadsdirect.co.uk/shop/schuko-cee77-plug-rewireable-black-right-angle/); equivalents at Reichelt/Conrad. Price ≈ **€3–6**.

| Spec | Real part | Spike model (`docking/schuko.py`) |
|---|---|---|
| Pin length | 19 mm | `PIN_LEN = 0.019` ✅ |
| Pin diameter | 4.8 mm | `R_PIN = 0.0024` ✅ |
| Pin centres | 19 mm | `PIN_SEP = 0.0095` ✅ |
| Body diameter | **36.7 mm** | `R_BODY = 0.01775` → 35.5 mm ⚠ |
| Rating | 16 A / 250 V | not simulated |

⚠ **Follow-up:** the real body is 36.7 mm, not the 35.5 mm the spike assumed. Against a
37 mm recess that leaves **0.15 mm clearance per side, not 0.75 mm** — a 5× tighter fit
than the tolerance sweep was run at. Confirm the figure on a specific part's datasheet
and re-run `scripts/schuko_spike.py`; the ±3 mm / ±3° envelope may shrink.

A right-angle plug is worth considering: it puts the cable exit parallel to the wall
instead of straight out along the arm axis.

### Compliant wrist (passive) — the accommodation that belongs on the robot

Not an actuated pivot. A **remote center compliance (RCC)**: a sprung mount between the
arm tip and the plug that lets the plug translate and rotate slightly, so contact forces
from a small misalignment *correct* the error instead of jamming. Published RCC devices
recover **up to ~4 mm and ~9°** of misalignment — comfortably more than our ±3 mm / ±3°
budget, and it works without any control loop.

- Build, don't buy: commercial RCC units are industrial-scale and priced accordingly. Four compression springs plus a floating plate around the plug body is the standard hobby equivalent. Budget **≈ €10** in springs and printed parts.
- Sim already models this: the spike's carrier uses 150 N/m lateral and 1 N·m/rad angular compliance. Those were guesses — measure the built part and update, since the whole tolerance envelope scales with them.

### Alignment feelers — ⚠ SLATED FOR REMOVAL (Ben, Aug 2026)

The feelers bake in an outlet-housing width that real outlets don't
standardize (multi-gang rectangles defeat the straddle); the circular well is
the only standard geometry, so the **well-centric plug-module redesign**
supersedes them. Already removed from the fork robot (`pluggybot_fork.xml`) —
at the hub they threaded between rack structures with mm margins, and the
rule is remove-not-design-around. The plug ROBOT keeps them until that
redesign so the milestone 6–7 measurements stay reproducible as recorded.

### Original feeler design (historical, still on the plug robot)

Two prongs on the lift carriage straddling the socket (lateral **±0.085 m**, on a
standoff bracket **2 cm above the plug axis**), tips 4 cm past the bumper. Printed
part + rubber tips, **≈ €2**. What they actually buy (measured): two-point wall
contact squares the robot's yaw mechanically — the axis the docking spike found
tightest — and gives a hard insertion-depth reference. What they do NOT buy: tipping
resistance (a wall contact cannot pull; the "brace" framing was falsified in sim).
Modelled in `pluggybot.xml` as `prong_l`/`prong_r`.

⚠ Both offsets are load-bearing, learned in the RL dock env (August 2026): the
original ±0.07 straddle cleared the ±0.042 *visual* plate but not the ±0.055
collision housing of the surface-mount socket — with >7 mm of lateral error a prong
tip landed on the housing edge 39 mm proud of the wall, wrecking the yaw squaring
and depth reference (every such episode stalled at the socket mouth). And at ±0.085
on the plug axis, the left prong swept through the battery's corner at low lift —
hence the 2 cm riser. Guarded by `test_prongs_clear_the_socket_housing` and the
geometric transit sweep in tests/test_arm.py.

### Docking camera (3rd camera)

**Raspberry Pi Camera Module 3** — a third unit, mounted on the *lift carriage* so it
rises with the plug. **€25,50** ([Welectron](https://www.welectron.com/Official-Raspberry-Pi-Camera-Module-3)).

Do **not** relocate the existing head pair: `perception/scanner.py` and
`mapping/occupancy_grid.py` bake in fixed camera offsets (0.03 forward / 0.03 left /
0.18 high), so a moving navigation eye would silently corrupt the map. Stretch splits
head and gripper cameras for the same reason. ⚠ The Pi 5 has only **two** CSI ports —
a third camera needs a CSI multiplexer/HAT or a USB camera instead. **TBD: resolve
before ordering.**

---

## Tool hub & modules (milestone 8) — PROVISIONAL, geometry validated in sim

The coupling spike (`scripts/hub_spike.py`) has validated the fork-and-peg
gravity latch: ±4 mm lateral / <2° yaw envelope, retention beyond the base's
traction limit, 300 g tool mass with margin. Parts below are the build list
that geometry implies; **prices/models TBD until the v2 geometry iteration
settles the yaw margin.**

| Part | Route | Notes → sim |
|---|---|---|
| Hub shelf + V-trays + back wall | **3D-printed** (PETG; the trays see ~3 N loads) | tray geometry = `hub/coupling.py` constants |
| Tool peg axles | **6 mm steel or alu rod**, cut to 150 mm (hardware store, ~€2) | printed pegs would flex/wear; the rod is the one loaded part |
| Arm fork + V-notches | 3D-printed, mounts where the plug's RCC sits (the plug becomes *a module*) | prong stance ±58 mm |
| Module frames (plug module, LCD module) | 3D-printed plates, common peg interface | ≤150 g budget each (validated to 300 g) |
| Module electronics | 1× ESP32-class board per module (~€5 each) | **power-only coupling, wireless data** — keeps the mating interface dumb and tolerant |
| Charge contacts | pogo-pin pairs (spring-loaded, ~€5) on the hub face, pads on the robot | gravity preload from the hang; the electrical-contact criterion carries over verbatim |
| Hub power | 12.6 V CC/CV charger board (3S, ~€10–15) fed by a mains adapter; balance leads handled robot-side by a 3S BMS | replaces wall-outlet charging as the primary path |
| LCD (first demo module) | small SPI/I2C display driven by the module's ESP32 | display-only; zero mechanical demands |

Open questions for the physical design: pogo-pin placement that engages by
the same hang motion (no extra alignment), whether the trays need steel wear
inserts, and the v2 yaw margin (chamfered trays / squaring press — see
SimNotes).

## Power

### Battery — 3S LiPo (recommended), part TBD

The motors are 12 V; a 3S LiPo is 11.1 V nominal / 12.6 V charged, which suits them
directly. 4S (14.8 V) would need a buck converter. LiFePO4 4S (12.8 V) is the safer,
heavier, longer-lived alternative and is worth pricing too.

| Spec | Target | Why |
|---|---|---|
| Chemistry / cells | 3S LiPo (11.1 V) | matches the 12 V gearmotors without regulation |
| Capacity | **≥ 5000 mAh** | 2× 5.5 A stall motors + Pi 5 (~5 A @ 5 V through a buck) |
| Continuous discharge | ≥ 20 C | headroom over the ~11 A both-motors-stalled worst case |
| Connector | XT60 | standard, and matches the Cytron MDD10A wiring |
| **Mass** | **~400 g** | **a feature, not a cost — it is the traction ballast (see mass budget)** |
| Mount position | **~x = +0.05, as low as possible** | traction 7.1 → 8.2 N, tipping 2.5 → 4.2 N |

Widely available (Gens Ace, Ovonic and similar in this size); **TBD: pick a specific
pack from a German retailer and confirm physical dimensions against the chassis plate.**
Also needed: a 3S balance charger, and a **12 V → 5 V 5 A buck converter** for the Pi 5.

⚠ Adding ~400 g takes the robot from 1.14 kg to ~1.54 kg. That invalidates the current
`pluggybot.xml` mass properties and every physics regression threshold derived from
them — re-run the suite after the model update, and expect to re-tune.

## Electronics — later (low priority)

- **Motor driver:** [Cytron MDD10A](https://botland.store/drivers-for-dc-motors/15818-cytron-mdd10a-dual-channel-30v-10a-motor-controller-5904422350444.html) (Botland, price TBD ~€20-class) — dual channel, 10 A continuous / 30 A peak per channel at 5–30 V, comfortably covers the 5.5 A stall current per motor.
- **Compute:** Raspberry Pi 5 8 GB — [BerryBase](https://www.berrybase.de/en/raspberry-pi-5-8gb-ram); ⚠ cheapest Geizhals listing July 2026 is **€202,90 incl. active-cooler kit** ([Geizhals](https://geizhals.de/raspberry-pi-5-modell-b-a3096144.html)) — well above the historical ~€90 board price (RAM price surge); verify standalone-board price before budgeting.

---

## Open decisions

1. ~~Stereo camera~~ → **Decided (July 2026): 2× Camera Module 3** with a custom rigid 60 mm bracket. Rationale: 4× cheaper, exact baseline control, 41° fovy (applied to sim), and the OAK-D Lite's ~35 cm minimum depth is a bad fit for terminal docking approach.
2. ~~Wheel diameter~~ → **Decided (July 2026): 90×10 mm** (sim r = 0.045 applied).
3. ~~Gear ratio~~ → **Decided (July 2026): 50:1 (#4753)** — docking push force over top speed (sim updated; see Motors section).
4. Chassis material/supplier (Misumi/igus vs laser-cut) — blocked on motor-bracket choice.
5. ~~Dock forward or backward?~~ → **Decided (August 2026): forward.** Re-measured on
   the armed robot: forward's failure mode is a benign ~3 N slide (not the destructive
   tip first feared), backward buys only ~1 N more at the cost of a rear camera and a
   blind approach. The mast is placed for forward docking (`pluggybot.xml`).
6. **Battery pack** — specific 3S LiPo (or 4S LiFePO4) from a German retailer, with
   dimensions checked against the chassis plate. Position ~x = +0.05 and low.
7. **Third camera routing** — Pi 5 has two CSI ports and both are used. CSI
   multiplexer, or a USB camera for docking? Blocks the lift-carriage design.
8. **Plug body diameter** — 35.5 mm (spike assumption) vs 36.7 mm (spec found for
   rewireable CEE 7/7). Confirm on a real datasheet, then re-run `schuko_spike.py`;
   the docking tolerance envelope depends on it.
9. **Lift/arm stroke + price** — get an igus quote for two NEMA11 lead-screw actuators
   (~0.25 m and ~0.20 m stroke) and their masses.
