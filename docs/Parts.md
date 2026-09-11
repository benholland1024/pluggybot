# Parts List

Specs and sourcing for the real parts the sim is modelled on, and the sim
parameter each number feeds. Hardware honesty (PluggyPlan.md, "What stays
fixed") is why this file exists: every part is purchasable, and a sim
constant with no part behind it is a guess and is marked as one.

> **Sourcing note:** Pololu (US) parts are stocked by German/EU distributors — mainly [Eckstein-shop.de](https://eckstein-shop.de/Pololu_EN), plus BerryBase, EXP-Tech, Welectron, Botland and TME.eu — so no US import is needed. All prices below are **approximate, incl. 19% VAT, as of July 2026** — re-check before ordering.

---

## Scale — how big is any of this?

**Measured off the models** (`room_hub.xml`, arm stowed), because the numbers
are easy to misread: MJCF `size` values are **half**-extents, so the chassis
line `size="0.12 0.09 0.03"` means a **24 × 18 × 6 cm** slab, not 12 × 9 × 3.

| | Footprint | Height |
|---|---|---|
| **Robot** (over wheels, mast up) | 24 cm long × 22 cm wide | **50 cm** |
| — chassis slab alone | 24 × 18 cm | 6 cm thick |
| — track width (wheel centres) | 21 cm | wheels ⌀90 mm |
| — battery bay | 13 × 4.6 × 2.6 cm | matches a real 5000 mAh 3S pack |
| **Tool rack** | 189 cm wide × 17 cm deep | **55 cm** |

So: a robot roughly the size of a shoebox with a 50 cm mast, **2.44 kg** in
the model (fork robot with LIDAR and arm; the plug robot is 2.34 kg), and a
rack as wide as a bookshelf: five tool bays at 0.25 m pitch plus the charge
bay is 1.86 m of rail (`rack/coupling.py` `RACK_HALF_W` = 0.93, re-checked
against both rooms it stands in — ToolPattern.md §6).

**Build medium.** The rack does NOT fit a typical 220 × 220 mm print bed, and
should not try to: print the *brackets, V-trays, tag plates and fork* (all
small, all tolerance-critical), and make the **rail, posts and base from
stock material** — wood is entirely reasonable, as is 2020 aluminium
extrusion if you want the bay pitch to be adjustable later. Only the parts
that touch the peg need print accuracy.

**Electronics volume.** A Pi 5 is 85 × 56 mm; the 24 × 18 cm chassis has room
for Pi + motor driver + battery without growing. No accelerator HAT is
needed (PluggyPlan.md "Road to hardware", item 1).

## Drive system

### Motors (2×) — ✅ CHOSEN: Pololu 50:1 (July 2026)

**Pololu 50:1 Metal Gearmotor 37Dx70L mm 12V with 64 CPR Encoder (Helical Pinion)** — Pololu #4753

- Source: [Eckstein-shop.de](https://eckstein-shop.de/Pololu-501-Metal-Gearmotor-37Dx70L-mm-12V-with-64CPR-EncoderHelical-Pinion-EN) — **€84,43 each**. Datasheet: [pololu.com/product/4753](https://www.pololu.com/product/4753)

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| Gear ratio | 50:1 | wheel-joint `armature="0.012"` (reflected rotor inertia ∝ ratio²) and `damping="0.05"` (the gearbox's ~30–35 % torque loss, per Pololu's ~65 % efficiency). Both load-bearing for sim stability — SimNotes "Physics modeling rules" |
| No-load speed @ 12 V | 200 rpm = **20.9 rad/s** | actuator `ctrlrange` ±21; `power.NOLOAD_SPEED` |
| Stall torque @ 12 V | 21 kg·cm = **2.06 N·m** | actuator `forcerange` ±2.06; `power.STALL_TORQUE` |
| Stall current @ 12 V | 5.5 A | `power.STALL_A` (brushed-DC current ∝ torque); motor driver sizing |
| No-load current | 0.2 A | `power.NOLOAD_A` |
| Encoder | 64 CPR motor shaft = **3200 CPR at output** | future encoder-quantization model (odometry is perfect-encoder today) |
| Mass | **205 g** | motor `geom` mass |
| Output shaft | 16 mm long, **6 mm D-shaft** | wheel/hub compatibility (below) |
| Dimensions | ⌀37 × 70 mm (excl. shaft) | motor geom size |

Runner-up, not selected: the 30:1 sibling (#4752, same price; 330 rpm /
1.37 N·m) — passed over for the 50:1's push force, because speed is a low
priority for this robot.

### Wheels (2×)

⚠ **Finding:** Pololu's 60–70 mm wheels only fit 3 mm shafts. For the 37D's **6 mm D-shaft**, the verified Pololu path is a 90 mm (or 80 mm) wheel + universal mounting hub.

| Part | Source | Price | Notes |
|---|---|---|---|
| **Pololu Wheel 90×10 mm pair** (#1435–1439) | [Eckstein-shop.de](https://eckstein-shop.de/Pololu-Wheel-90x10mm-Pair-Red-for-Micro-Metal-Gearmotors-EN) | **€11,13 / pair** | Six M3/#4-40 mounting holes matching Pololu universal hubs. Mass: TBD (verify on datasheet) |
| **Pololu Universal Aluminum Mounting Hub, 6 mm shaft, M3 holes (2-pack)** (#1999) | [Eckstein-shop.de](https://eckstein-shop.de/PololuUniversalAluminumMountingHubfor6mmShaft2CM3Holes2-PackEN) | **€11,95 / 2-pack** | Set-screw hub for the 6 mm D-shaft; wheel bolts to hub |
| Alt: Pololu Multi-Hub Wheel 80×10 mm (2-pack) | [Eckstein-shop.de](https://eckstein-shop.de/Pololu-Multi-Hub-Wheel-w-Inserts-for-3mm-and-4mm-Shafts-8010mm-Black-2-pack-EN) | €14,20 / 2-pack | Inserts are 3/4 mm only — **TBD: verify it accepts the 6 mm universal hub** before buying |

✅ **Decided (July 2026): 90×10 mm** → wheel geom r = 0.045 m
(`control.WHEEL_RADIUS`), half-width 0.005. With the 50:1 motor, top speed
is 20.9 × 0.045 ≈ **0.94 m/s** and stall push ≈ 2.06/0.045 ≈ 46 N per wheel.
The larger radius destabilised the sim's pitch dynamics until tyre
compliance (`solref="0.05 1"`) and gearbox damping were modelled — SimNotes.

---

## Chassis & mechanical

| Part | Source | Price | Specs → sim |
|---|---|---|---|
| **Pololu Ball Caster with 3/4″ metal ball** (#955) | [EXP-Tech](https://www.exp-tech.de/zubehoer/mechanische-bauteile/5551/pololu-ball-caster-with-3/4-metal-ball) (in stock); also [BerryBase](https://www.berrybase.at/pololu-ball-caster-0-75-zoll-metallkugel-abs-gehaeuse-hoehenverstellbar-fuer-kleine-roboter), [TME.eu](https://www.tme.eu/en/details/pololu-955/accessories-for-robotics-and-rc/pololu/ball-caster-with-3-4-metal-ball/) | **≈ €4,50** (€3,72 net) | Ball ⌀ 19 mm → caster `sphere` geom; height 0.83″–≈1″ (21–25 mm) adjustable via spacers → chassis ground clearance. Mass: TBD (verify on datasheet). ⚠ Its low friction needs `priority="1"` or MuJoCo takes the pair MAX — SimNotes "THE caster lesson" |
| Pololu 37D Metal Gearmotor Bracket (pair) | [Eckstein Pololu mounts category](https://eckstein-shop.de/Pololu-Motor-Mounts-Wheel-EN) | TBD | Sets motor axle height above chassis plate |
| Chassis plate | TBD (laser-cut acrylic/alu, or Misumi/igus stock profiles) | TBD | Track width target 0.21 m — set by motor bracket spacing, verify wheel-to-wheel once brackets chosen |
| **Front bumper switch** (issue #94) | a sprung bumper bar over 1–2 lever microswitches (e.g. Omron D2F / any "roller lever" micro, ~€1 each) on the chassis' front face, 6–12 cm off the floor; a rear one is optional | **≈ €2–5** | Feeds `HubSwap.pressing`: dead reckoning HOLDS its travel while the bumper is pressed on the side the wheels are rolling toward (wheel direction from the encoders). In the sim the "switch" is the chassis box's front face read off the contact list, reduced to front/back — nothing a €1 switch does not report. Why it exists: a drive stalled against a fence pumped 4.28 m of imaginary travel in 30 s, and motor torque cannot tell a stall from a cruise (0.44 vs 0.35 N·m — the tyres slip long before the motors saturate). The sim holds the press `PRESS_RELEASE_S` = 50 ms past the last contact because a rigid chassis bounces off a rigid fence; a sprung bar stays pressed through that. Blind spot, same on both: anything between the bumper's top (~12 cm) and the lidar plane (22 cm) meets the fork, not the bar. SimNotes "A stalled drive is an odometry pump" |

---

## Vision & ranging

### ✅ CHOSEN (Aug 2026): 2D LIDAR + ONE navigation camera — stereo dropped

**Decision.** The hub-era robot ranges with a scanning LIDAR and sees with a
single camera; the stereo pair is retired. Applied to `pluggybot_fork.xml`.
`pluggybot.xml` (plug robot) keeps its pair, frozen for milestone 6–7
reproducibility.

**Why, measured.** Real SGBM on this sim's own stereo pair, in the best case
a stereo pair could ever have, produced disparity for only **49.7 %** of the
mapper's scan row at **593 mm** median error mid-room, against a 50 mm grid
cell; geometry agrees (σ_z ±649 mm at 5 m on a 60 mm baseline). Stereo cannot
build this map: the room is flat painted walls, the classic no-disparity
case. Tables and method: SimNotes "Sensor-realism pass".

| Part | Source | Price | Specs → sim |
|---|---|---|---|
| **2D LIDAR** — Slamtec RPLIDAR C1 (or A1M8) | Botland / EXP-Tech / Welectron | **≈ €65–100** | 360°, 10 Hz, 12 m, ±30 mm, ~110 g, ~2.5 W over USB/UART. `perception/lidar.py`: 360 `mj_ray` casts at `LIDAR_PERIOD` 0.1 s, ±10 mm + 1 % noise, 2 % dropout, `max_range` 8 m, self-hits dropped. Its offset from the axle is `LIDAR_ORIGIN`, which the grid update bakes in — move the unit, move the constant |
| **Navigation camera** — 1× Pi Camera Module 3 | [Welectron](https://www.welectron.com/Official-Raspberry-Pi-Camera-Module-3) | €25,50 | IMX708, 66° × **41°** FOV → `fovy="41"`; PDAF ~10 cm–∞; 4 g. `left_eye` on the head: AprilTags (+ outlet detection if the plug module is ever built) |
| **Docking camera** — 1× Pi Camera Module 3 | as above | €25,50 | `dock_eye` on the lift carriage, so it rises with the fork; same optics |

**What it bought**
- Two cameras on the Pi 5's two CSI ports — no multiplexer (the old
  third-camera blocker, closed).
- Drops SGBM's ~60 ms/frame from the Pi 5 perception budget
  (~138 → ~78 ms per cycle).
- 360° coverage, where the forward ±20° safety reflex could never protect a
  spin (SimNotes, milestone 4); works in the dark and on textureless walls.

**What it costs**
- **~2.5 W continuous** → `power.ELECTRONICS_W` 6.0 → 8.5, straight off run
  time.
- **~110 g mounted high** (body-local z = 0.15) for a clear scan plane at
  0.223 m.
- **A ~17° blind sector behind-right** where the mast stands in the scan
  plane, centred at `atan2(−0.05, −0.10)`; self-hits are dropped rather than
  reported as free space (`tests/test_lidar.py` pins both).
- The outlet-landmark projection lost its depth source and would need a
  bearing + wall-intersection rewrite — plug-anywhere path only.

Superseded, for the record: **DIY stereo** (2× Camera Module 3 on a custom
60 mm bracket, July 2026) was the right pick against the **Luxonis OAK-D
Lite** (€193–199, 75 mm baseline, 35 cm minimum depth), but the question
neither asked was whether any stereo pair could produce the mapper's scan.

---

## Arm & docking (milestone 6)

### Naming

**Mast** = the fixed vertical column. **Lift** = the carriage that travels up
it (and the actuator driving that). **Telescoping arm** = the horizontal
extension carrying the plug, and now the fork. Hello Robot Stretch's
vocabulary and architecture: base owns x/yaw, lift owns z, arm owns reach.

### ⚠ The mass budget is the binding constraint

Measured in sim (`models/world.xml`, headless force probe, ramped loads on
the armed robot): pushing at outlet height (0.30 m), forward docking holds
**~3 N then slides**; backward holds ~4 N then goes caster-light. Schuko
insertion forces measured by the spike: **0.7 N** perfectly aligned,
**6.1 N** at 2 mm lateral, **7.8 N** at 2° yaw. So insertion must stay
≲3 N → terminal alignment ≲1 mm, a compliant wrist, and forward docking
(mapping and cameras face forward; the failure mode is a benign slide).

Two things that came out of it and still bind:
- **Battery position is a design variable, not packaging** — x = +0.05
  (ahead of centre) for tipping margin, **y = +0.06 as a counterweight** for
  the arm assembly hanging at y = −0.05, without which the robot veers 26 cm
  right over 4 m open-loop. `test_arm_mass_is_counterbalanced` pins it;
  carrying the heaviest module adds 5.5 mm over 2.7 m and needs no re-tune
  (SimNotes "Asymmetric mass makes a diff-drive veer open-loop").
- **A wall cannot brace you.** A wall contact is one-way — it pushes the
  robot the same way the insertion reaction does — so bracing pads were
  falsified in sim (SimNotes "A wall cannot brace you").

### Lift and telescoping arm — 2× linear actuator

**igus drylin® E lead-screw stepper linear actuator, NEMA11** — igus DLE-LA-0001

- Source: [igus.com/product/DLE-LA-0001](https://www.igus.com/product/DLE-LA-0001). Price: **TBD** — igus quotes stroke-configured units through their configurator, not a fixed list price. Get a quote for both axes together.

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| Max thrust | **50 N** | actuator `forcerange` ±50 — 6× the worst-case 7.8 N insertion |
| Holding torque | 0.12 N·m | holds position unpowered → `power.ACTUATOR_W` (5 W) is drawn only while moving, and a parked module axis is a position servo at its target (`rack/coupling.py`) |
| Lead screw | dryspin® high helix DST 6.35 × 5.08 | 5.08 mm travel per revolution |
| Linear feed per step | **0.0254 mm** (1.8° step) | far finer than the ±3 mm docking budget; positioning is not the limit |
| Motor flange | NEMA11 / 28 mm | mount design |
| Lubrication | none required (dryspin®) | — |
| Stroke | configurable — want **~0.25 m lift**, **~0.20 m reach** | joint `range` (the model's lift travels 0.31 m) |
| Mass | TBD (verify on datasheet) | body mass — matters, see mass budget above |

Stroke rationale: the lift must span outlet heights 0.26–0.38 m in
`room_1.xml` and carry the docking camera high enough to keep a 0.38 m
outlet in frame at close range (a fixed 0.18 m eye loses it below 0.40 m —
measured). Reach is set by parking the base at ~0.25 m; a 0.6 m cantilever
is unaffordable on this chassis.

### Plug

**Rewireable Schuko CEE 7/7 plug (Type F)** — e.g. [Leads Direct rewireable right-angle](https://leadsdirect.co.uk/shop/schuko-cee77-plug-rewireable-black-right-angle/); equivalents at Reichelt/Conrad. Price ≈ **€3–6**. A right-angle plug puts the cable exit parallel to the wall instead of along the arm axis.

| Spec | Real part | Spike model (`docking/schuko.py`) |
|---|---|---|
| Pin length | 19 mm | `PIN_LEN = 0.019` ✅ |
| Pin diameter | 4.8 mm | `R_PIN = 0.0024` ✅ |
| Pin centres | 19 mm | `PIN_SEP = 0.0095` ✅ |
| Body diameter | **36.7 mm** | `R_BODY = 0.01775` → 35.5 mm ⚠ |
| Rating | 16 A / 250 V | not simulated |

⚠ **Open (decision 8):** the real body is 36.7 mm, not the 35.5 mm the spike
assumed. Against a 37 mm recess that is **0.15 mm clearance per side, not
0.75 mm** — a 5× tighter fit than the tolerance sweep was run at. Confirm on
a specific datasheet and re-run `scripts/schuko_spike.py`; the ±3 mm / ±3°
envelope may shrink.

### Compliant wrist (passive)

A **remote center compliance (RCC)**: a sprung mount between the arm tip and
the tool that lets it translate and rotate slightly, so contact forces from a
small misalignment *correct* the error instead of jamming. Published RCC
devices recover up to ~4 mm and ~9° — more than the ±3 mm / ±3° budget, with
no control loop. Build, don't buy: four compression springs plus a floating
plate, **≈ €10**. The sim models it as 150 N/m lateral and 1 N·m/rad angular
(`coupling.LAT_STIFFNESS` / `YAW_STIFFNESS`, and the fork model's wrist
joints); those were guesses — measure the built part and update, since the
whole tolerance envelope scales with them.

### Alignment feelers — ❌ removed from the hub robot

`pluggybot_fork.xml` has none (its `fork_prong_l/r` are the fork's tines).
The plug robot `pluggybot.xml` keeps them as `prong_l`/`prong_r`, frozen for
milestone 6–7 reproducibility: two prongs on the lift carriage straddling
the socket at lateral **±0.085 m**, on a bracket **2 cm above the plug
axis** — both offsets load-bearing (the ±0.07 they replaced landed on the
socket housing's edge; on the plug axis the left prong swept the battery),
`test_prongs_clear_the_socket_housing`. What they buy, measured: two-point
wall contact squares yaw and references insertion depth — not tipping
resistance. Why removed: they bake in an outlet-housing width real outlets
do not standardise; the circular well is the only standard geometry, so a
plug-anywhere module, if it is ever built, is well-centric.

---

## Tool hub & modules (milestone 8)

The coupling spike (`scripts/hub_spike.py`) validated the fork-and-peg
gravity latch: ±4 mm lateral / <2° yaw envelope, retention beyond the base's
traction limit, 300 g tool mass with margin (ToolPattern.md §2 is the
envelope). Parts below are the build list that geometry implies; **prices
and models TBD.**

| Part | Route | Notes → sim |
|---|---|---|
| Rack: rail, posts, shelf, V-trays, wall braces | rail/posts/base from stock (above); trays and brackets **3D-printed** (PETG; the trays see ~3 N loads) | geometry = `rack/coupling.py` constants; 1.86 m of rail for five bays + charge bay |
| Tool peg axles | **6 mm steel rod** (conductive — see below), 2× 63 mm conductors on a 24 mm insulating centre bush, 150 mm overall | the one loaded part **and the electrical connector** — `PEG_R`, `PEG_HALF`, `PEG_INSUL_HALF` / `PEG_COND_HALF` |
| Arm fork + V-notches | 3D-printed, mounts where the plug's RCC sits | prong stance ±58 mm (`FORK_Y`) |
| Module frames (LCD, plug, pen, claw, seed dispenser) | 3D-printed plates, common peg interface | 130–211 g measured, ~250 g practical ceiling (ToolPattern.md "Mass and geometry class") |
| Module electronics | 1× ESP32-class board per module (~€5 each) | **power-only coupling, wireless data** — keeps the mating interface dumb and tolerant. Modelled as a 0.6 W load (`power.MODULE_IDLE_W`) drawn only while the coupling conducts |
| **Module power contacts** | **none to buy — the peg IS the connector** | Split peg + the fork's two V-notch pairs = a two-pole coupling with 0.43–0.47 N of gravity preload per plate, already there, self-wiping on the seating slide. Needs a conductive rod, an insulating centre bush, isolated V-plates, and a **holding capacitor sized for ~200 ms** — measured worst outage 178 ms under hard driving with honest peg friction (SimNotes "The module electrical interface"; ToolPattern.md §3) |
| Charge contacts | pogo-pin pairs (spring-loaded, ~€5) on the hub face at bumper height (`CHARGE_PIN_Z` 0.09), pads on the robot | preload from the drive-in press; the electrical-contact criterion carries over verbatim |
| Hub power | 12.6 V CC/CV charger board (3S, ~€10–15) fed by a mains adapter; balance leads handled robot-side by a 3S BMS | replaces wall-outlet charging as the primary path |
| LCD module | small SPI/I2C display driven by the module's ESP32 | display-only; the face is drawn in the browser off a streamed enum |

Open for the physical design: pogo-pin placement that engages by the same
drive-in motion (no extra alignment), and whether the trays need steel wear
inserts.

## Power

### Battery — 3S LiPo (recommended), part TBD

The motors are 12 V; a 3S LiPo is 11.1 V nominal / 12.6 V charged, which suits
them directly. 4S (14.8 V) would need a buck converter. LiFePO4 4S (12.8 V) is
the safer, heavier, longer-lived alternative and is worth pricing too.

| Spec | Target | Why / → sim |
|---|---|---|
| Chemistry / cells | 3S LiPo (11.1 V) | matches the 12 V gearmotors without regulation → `power.NOMINAL_V` |
| Capacity | **≥ 5000 mAh** (~55 Wh) | 2× 5.5 A stall motors + Pi 5 (~5 A @ 5 V through a buck). `power.CHARGE_W` 55 W is ~1C into it; capacity in the sim is a knob (demo and hosting cells), `--battery-wh 55.5` runs the real pack |
| Continuous discharge | ≥ 20 C | headroom over the ~11 A both-motors-stalled worst case |
| Connector | XT60 | standard, and matches the Cytron MDD10A wiring |
| **Mass** | **~400 g** | **a feature, not a cost — it is the traction ballast** (mass budget above) |
| Mount position | **x = +0.05, y = +0.06, as low as possible** | traction 7.1 → 8.2 N, tipping 2.5 → 4.2 N; the y offset is the counterweight |

Widely available (Gens Ace, Ovonic and similar in this size); **TBD: pick a
specific pack from a German retailer and confirm physical dimensions against
the chassis plate.** Also needed: a 3S balance charger, and a **12 V → 5 V
5 A buck converter** for the Pi 5.

⚠ The models already carry the pack as a 400 g placeholder box at that
position. A real pack of a different mass or footprint moves every physics
threshold derived from the model — the mass re-budget in PluggyPlan.md
"Road to hardware" (item 5), to be done last.

## Electronics — later (low priority)

- **Motor driver:** [Cytron MDD10A](https://botland.store/drivers-for-dc-motors/15818-cytron-mdd10a-dual-channel-30v-10a-motor-controller-5904422350444.html) (Botland, price TBD ~€20-class) — dual channel, 10 A continuous / 30 A peak per channel at 5–30 V, comfortably covers the 5.5 A stall current per motor.
- **Compute:** Raspberry Pi 5 8 GB — [BerryBase](https://www.berrybase.de/en/raspberry-pi-5-8gb-ram); ⚠ cheapest Geizhals listing July 2026 is **€202,90 incl. active-cooler kit** ([Geizhals](https://geizhals.de/raspberry-pi-5-modell-b-a3096144.html)) — well above the historical ~€90 board price (RAM price surge); verify standalone-board price before budgeting.

---

## Open decisions

1. ~~Stereo camera~~ → **Superseded (Aug 2026): 2D LIDAR + one camera** — "Vision & ranging".
2. ~~Wheel diameter~~ → **Decided (July 2026): 90×10 mm** (sim r = 0.045).
3. ~~Gear ratio~~ → **Decided (July 2026): 50:1 (#4753)** — push force over top speed.
4. Chassis material/supplier (Misumi/igus vs laser-cut) — blocked on motor-bracket choice.
5. ~~Dock forward or backward?~~ → **Decided (Aug 2026): forward** — a benign ~3 N slide, and the cameras face that way ("The mass budget").
6. **Battery pack** — specific 3S LiPo (or 4S LiFePO4) from a German retailer, with
   dimensions checked against the chassis plate. Position x = +0.05, y = +0.06, low.
7. ~~Third camera routing~~ → **Closed (Aug 2026) by the LIDAR swap**: two cameras on two CSI ports, LIDAR on USB/UART.
8. **Plug body diameter** — 35.5 mm (spike assumption) vs 36.7 mm (spec found for
   rewireable CEE 7/7). Confirm on a real datasheet, then re-run `schuko_spike.py`;
   the docking tolerance envelope depends on it.
9. **Lift/arm stroke + price** — get an igus quote for two NEMA11 lead-screw actuators
   (~0.25 m and ~0.20 m stroke) and their masses.
