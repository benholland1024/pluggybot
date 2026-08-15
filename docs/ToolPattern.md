# Tool Pattern — how to build the next module

The recipe for adding a tool to PluggyBot's hub, extracted retrospectively
from the two that already work: the **pen** (an X-Y plotter, `hub/drawing.py`)
and the **claw** (a pick-and-place gripper, `hub/gripper.py`). Both were built
the same way, and the sequence is written down here so tool #3 onward costs a
fraction of what tool #1 did.

**Validated once, by building against it.** The **seed dispenser**
(`hub/dispenser.py`, the fifth tool) was built from this doc rather than mined
for it. Stage 0 came out entirely on paper and was right, and the mechanism
worked on the first run. Four gaps it exposed have been folded back in and are
marked ⓘ **found by building the dispenser** where they appear: the *tolerance
class* question in §5.0, the *payload* row in §2 and §3, contact rule 3
(rolling resistance), and the telemetry entry in §5.6. Keep doing this — the
next tool's gaps belong here too.

Read this alongside, not instead of:
- `docs/SimNotes.md` — the war stories behind every number quoted here. This
  doc says *what the constraint is*; SimNotes says *how we found out*.
- `docs/Parts.md` — the hardware decisions the sim parameters are modelling.
- `CLAUDE.md` — the house rules (ramping, solver policy, the `priority`
  gotcha) in their short form.

---

## 1. What a tool module is

**A tool is not cargo. It is a kinematic extension the robot acquires by
picking it up.** That framing is the whole design, and it is what makes each
tool cheap: the robot already owns three axes, and the module only has to
supply the ones it lacks.

| axis | who owns it | notes |
|---|---|---|
| x, yaw | the differential-drive base | nonholonomic — no lateral translation |
| z | the prismatic lift | 0–0.31 m of travel |
| reach (approach depth) | the arm | 0–0.20 m; also sets contact *pressure* |
| **anything else** | **the module** | this is what you are designing |

The pen module supplies lateral travel (a carriage along the peg axis) —
precisely the axis a differential drive structurally cannot produce — and
pairs it with the robot's lift to make an X-Y plotter. The claw supplies a
grip, and lets the robot's lift be its up-and-down, so the claw itself only
opens and closes. Neither module reimplements an axis the robot already has.

Before designing anything, answer: **which axis does this tool bring, and
which does it borrow?** A tool that needs an axis the robot has *and* the
module has is a design that has not been decomposed yet.

The seed dispenser is the cleanest case: what it brings is a **discrete
release** — the robot can carry and can grip, but it cannot let go of exactly
one of something — and what it borrows is the lift (drop height) and the base
(placement). No new axis at all, which is most of why it was cheap.

The module also gets, for free:
- **power**, through the coupling itself (§3)
- **data**, wirelessly — module signals do not cross the coupling, so a
  module-mounted camera or sensor costs no port on the robot's own compute
  (the claw's `claw_eye` is the first camera on a tool rather than a chassis)
- **a fiducial identity**, from its own AprilTag

---

## 2. The coupling interface envelope

The mechanical contract. These numbers are measured, they are pinned by
`tests/test_hub_coupling.py`, and a new tool does not get to renegotiate
them — it gets to fit inside them.

### The latch

A tool hangs by a **long horizontal peg axle** (6 mm steel rod, 150 mm long:
`PEG_R = 0.003`, `PEG_HALF = 0.075`) resting in two upward-open **V-trays** on
the rack. The robot's arm carries a **fork** whose two prongs are tipped with
matching V-notches, and it grabs the peg *outboard* of the trays — so lateral
capture is set by how far the peg overhangs, not by machined clearance.

The verbs are **slide in, lift, back away** (and the reverse to return),
because the robot has no wrist. **Gravity is the latch.** There is no spring,
magnet, catch, or actuator anywhere in the coupling.

### Alignment envelope

| axis | envelope | notes |
|---|---|---|
| lateral (y) | **± 4 mm** | full pick-and-return cycle |
| along-track (x) | **± 11 mm** | the approach-depth capture window |
| vertical (z) | **−8 / +6 mm** | |
| yaw | **< 2°** | the tight axis — picks survive 2°, **returns do not**; ±4° jams at 50–120 N |

Yaw being the binding constraint is not a surprise; the schuko docking spike
found the same thing about its own interface. Navigation delivers ~0.5° of
settle, so v1 is usable, but the margin is thinner than it should be.

**Retention is not the problem.** A carried tool holds through 8 m/s² of shake
— more than the wheels can transmit to it — and the spike carried a 300 g tool,
2× the nominal per-module cap.

### The moment budget — the number that shapes tools

**The gravity latch takes about 0.45 N·m of pitch moment before the peg rides
out of its V.** This is the single most important constraint on a tool's
*shape*, and it is a moment limit, not a mass limit:

| forward reach L | 200 g | 400 g | 800 g |
|---|---|---|---|
| 0 mm | ok | ok | **ok** |
| 100 mm | ok | ok | unseats |
| 150 mm | ok | **unseats** | — |

**Reach is far more expensive than mass.** 800 g hangs happily on the peg's
own axis; 400 g at 150 mm out unseats the module. This is why the claw is a
pendant straight *down* the peg axis rather than the angled arm the idea
started as, and why its eventual 55 mm of forward angle was affordable (60 g
at 55 mm is 0.032 N·m — 7 % of the budget).

The chassis, incidentally, is nowhere near its own limits: 800 g at the peg
drops drive-wheel load only 15.0 → 12.6 N, and static forward tipping needs
about 5 kg. The obvious worry was the wrong one.

### The force budget — what the lean-pad buys

**A gravity latch cannot push.** The restoring torque is `m·g·d·sin θ` with
d = `PEG_ABOVE_BODY` = 22 mm, and a horizontal tool force has *the same*
22 mm moment arm about the peg. Equal arms means any tool force buys a
proportional, large rotation — and it runs away past ~0.25 N, because
rotating swings the tip further out and grows the disturbing arm faster than
`sin θ` grows the restoring one. Bare, the latch gives out at **0.1 N**.

The **lean-pad** on the fork (`pluggybot_fork.xml`) reacts that torque in
compression at a ~29 mm lever instead of by gravity at 22 mm:

| tool force | without pad | with pad |
|---|---|---|
| 0.10 N | −5.30° / 2.10 mm tip retreat | −0.01° / 0.01 mm |
| 0.50 N | −51.32° / 38.27 mm (flopped) | −0.06° / 0.07 mm |
| 2.00 N | −52.51° / 39.50 mm | −0.19° / 0.26 mm |

So: **a tool may exert up to ~1.5–2 N against the world.** Past that the peg
rides up its own V-notch and out. The response is linear (~0.095°/N), so a
tool that needs 5 N is not a tuning problem — it is a different coupling.

Convenient sign, worth knowing when you place a tool's business end: the
reaction rotates the module's lower back *toward* the robot, i.e. into the
pad. Drawing load and pad preload push the same way. A tool that pushed the
other way would be fighting the pad, not using it.

### Mass and geometry class

New modules must stay in the existing class or the lift preset and RCC-wrist
tuning reopen. Measured, in `hub_world.xml`:

| module | mass |
|---|---|
| `module_lcd` | 130 g |
| `module_plug` | 156 g |
| `module_pen` | 182 g |
| `module_seed` | 186 g |
| `module_claw` | 211 g |

`module_xml(mass=0.12)` is the **plate + peg budget**; whatever the `face`
adds sits on top of it. Treat ~250 g as the practical ceiling until someone
re-measures the lift preset, `DROOP_COMP` and the pad geometry against a
heavier tool.

Geometry limits that have each already broken something:

- **80 mm** between a racked module's front face and the wall. Modules hang
  business-end-inward, so *whatever your tool sticks out the front points at
  the wall when stowed*. This is what bounds the claw's forward angle — the
  rack, not the moment budget.
- **The bracket band: z −30…−9 mm of the module's own frame is the rack's,
  not yours.** ⓘ *Found by issue #10.* A set-down needs the module raised
  **31 mm** — `RETURN_CLEARANCE` (20) plus the ~11 mm a carried peg already
  rides above its hung rest — so anything your tool puts in that band
  arrives under the bay's tray brackets and jams. Standing proud in **x** is
  fine; the rack leaves that air empty. It is **height** that bites, which
  is the same thing `-TOOL_HALF_Z` above is telling you. If a part must live
  there, run a bare-world clearance sweep (raise the module 1 mm at a time
  at its bay and count real contacts — `mj_geomDistance` will *not* tell you,
  it does not measure box-box separation) *before* wiring the tool into a
  mission.
- **The parked envelope.** Arm retracted, the fork tucks over the chassis with
  only ~60 mm between the chassis top (z = 0.12) and the LIDAR's centre row
  (z = 0.18). A module part that hangs low enough to sit on the chassis jacks
  the whole fork up into the scan row.

### What a carried module does in transit

Measured on a carried module over a net-zero maneuver:

- **The hang is repeatable**: it returns to its resting lean within
  +0.00° / 0.02 mm, and settles under 0.5 mm in 0.14 s. So a tool tip *can*
  be calibrated against the resting pose.
- **In transit it swings 10.3° peak-to-peak** during a turn (±10.9 mm
  vertical at a pen-tip-length lever). That is a *sequencing* constraint, not
  a mechanical one: the business end must be clear of the world while
  driving, and **carry clearance is sized by swing, not by static height**
  (46 mm of "clear of the floor" was not clear at all — a 172 mm pendant
  swinging 10° moves its tip ~30 mm, and the carried block struck the floor
  mid-turn and was knocked out of the jaws).

### If the tool has a moving axis, it needs a STOW POSE

ⓘ *Found by issue #10.* A tool that brings its own axis can be *left*
anywhere on that axis when the job ends, and "wherever the last stroke
finished" is not a pose anyone designed. The pen's carriage ends a square at
**+37 mm**, and there it sits in the y band where the bay's tray brackets
hang instead of in the gap between them — clear window 15–27 mm against a
stow that needs 31. Centred, the window is 15–39 mm.

So: **decide where your axis parks for a stow, and return it there as part
of putting the tool away** — `PenPlotter.carry_config` does this alongside
restoring the lift, and for the same reason. Two things make this trap
expensive if you leave it for later:

- **It is invisible to every test that does not run the job first.** A pick
  and a return with no drawing in between passed in both worlds while
  `home_draw.py` still failed. If your fetch/stow test never actuates the
  tool, it is not testing the stow your mission does.
- **It fails as a clean-looking approach.** The module jams, the wheels
  slip, dead reckoning counts the slip as progress, and `_drive_until`
  returns **"arrived"** having stopped 25–31 mm short. Nothing in the
  report says "obstructed".

### If the tool carries a payload

ⓘ *Found by building the dispenser.* Some tools act **on** the world (pen,
claw); some **carry** things and let them go. If yours carries, decide which
kind of retention it uses, because they cost completely different amounts of
analysis:

- **Retained by grip** (the claw's block) — the hold is a force, so it can be
  lost. This is what needs the swing analysis above, a clamp-force choice, a
  hard `solimp` on the pads, and a contact criterion checked continuously.
- **Retained by geometry** (the dispenser's seeds: a capped tube over a shelf
  whose only exit is blocked by the shuttle) — there is no pose in which the
  payload has anywhere to go. No swing analysis, no clamp force, nothing to
  tune. **Prefer this whenever the payload does not need to be grasped.**

Payload mass counts against the moment budget exactly like structure does, at
whatever offset it sits — so a magazine on the peg's own axis is free, and a
hopper slung forward is not.

---

## 3. Module anatomy

Everything below lives in `src/pluggybot/hub/coupling.py`. A module is a free
body plus a string of extra geoms (`face`) injected into its frame.

### Required, and generated for you

`module_xml(name, x, y, peg_z, rgba, mass, face, yaw_deg)` emits:

- a **`<freejoint/>` body** at the hang pose — the body's z is *derived* from
  the peg height, tray drop and `PEG_ABOVE_BODY`, so never hand-place it;
- the **plate**: a 20 × 40 × 60 mm box (`TOOL_HALF_X/Y/Z` = 0.010/0.020/0.030);
- the **peg**, from `peg_xml()` — and the peg is where the interesting part is.

### The peg is also the electrical connector

`peg_xml()` splits the rod into **two conductors around an insulated centre**
(`PEG_COND_HALF` either side of `PEG_INSUL_HALF` = 12 mm). The fork's left
V-notch pair and right V-notch pair become the two poles of a power-only
coupling.

This costs nothing, which is the point: the peg already sits in four V-notch
plates carrying 0.43–0.47 N each of gravity preload (1.79 N total, summing to
the module's weight), the seating slide wipes the contacts clean, and the peg
was already the one metal part in the design. The alignment problem was
already solved by the latch. (A lean-pad's preload, by contrast, is capped at
~0.56 N absolute and realistically ~0.1 N — well under what a pogo pin needs.
That correction is recorded against Parts.md in SimNotes.)

Two things follow for every new tool:

- **`PEG_FRICTION = 0.4` with `priority="1"`.** MuJoCo's default μ = 1.0 has a
  45° friction angle — exactly the V's flank angle — so the peg sat right on
  the sliding threshold and barely self-centred. And without `priority`,
  setting a low friction does *nothing*, because MuJoCo combines pair friction
  as the elementwise **MAX**. Do not copy a peg without copying both.
- **`module_power_state(model, data, name)` is your seating check**, and it
  reports the poles **separately**. A half-seated coupling — on the fork but
  one pole open — is a real failure mode of a two-point latch, and a bare
  boolean files it under "off" with no way to tell a missing tool from a bad
  seat.

### The `face`: your tool's actual parts

Add a `<tool>_face` string to `_module_faces()` and return it. Everything in
it is expressed in the **module's own frame** (+x is the face toward the
robot; −x points at the wall when stowed). Elements a tool may carry:

| element | pen | claw | dispenser | notes |
|---|---|---|---|---|
| identity tag | ✔ | ✔ | ✔ | **required** — 30 mm tag36h11 on the +x face |
| passive structure | rail | pendant tube | magazine + shelf | mass counts against §2 |
| joints | 2 (carriage, quill) | 2 (jaw slides) | 1 (shuttle slide) | |
| actuators | 1 (the quill is a passive spring) | 2 (both jaws) | 1 | |
| working-point site | `pen_tip` | `claw_grip` | `seed_outlet` | what the controller reads |
| camera | — | `claw_eye` | — | aim with `_look_at()`, never hand-typed cosines |
| payload | — | grip-retained | geometry-retained | see §2 |

Notes drawn from all three builds:

- **Joints ≠ actuators.** The pen's quill is a *sprung* slide with no motor:
  60 N/m and 20 mm of travel, which turns pen pressure from a positioning
  problem into a design constant (~0.6 N at nominal press, varying only
  0.4–0.8 N across a whole figure). A first attempt at 200 N/m held ~0.5 mm of
  deflection and the pen lifted clean off the top of the figure, because the
  arm's droop *grows* with lift height. **Compliance where the tool meets the
  world is usually cheaper than accuracy in the arm.**
- **Model the actuator you would actually buy.** The pen carriage is
  `kp=2000` because it is a lead screw and a screw does not yield to drag
  (`kp=120` needed ~8 mm of tracking error to make 1 N, and the figure came
  out 12 mm RMS off). The claw is `kp=600` because that is an MG996R-class
  servo rather than a micro one — `kp=200` held a static lift and lost the
  block during a *turn*. Both are part choices, not tuning knobs.
- **A lead screw holds position unpowered**, which is why a position servo
  parked at its target is the honest model of a module sitting unpowered on
  the rack. If your tool's axis would *not* hold unpowered, model that.
- **Mount things in front of the plate, not inside it.** The pen's rail and
  carriage were first placed at the plate's own x, burying them in it — and
  MuJoCo only filters *parent-child* contact pairs, so the grandchild quill
  collided with the plate's corner, jammed at +21.9 mm, and stopped tracking
  for most of a figure. The clearance sweep missed it because it checked the
  pen against the *robot*, never against the module's own frame.
  The flip side is worth knowing too: a **direct child** body *is* filtered,
  so a moving part one level down needs no clearance against the structure it
  slides in. The dispenser's escapement shuttle runs inside the magazine with
  its faces flush against the tube and the shelf, and never touches them.
  Grandchildren get no such favour.
- **The structure carrying a tool must not occupy the tool's working space.**
  The claw's drop tube ran 28 mm into the grip zone: descending, it reached
  the block first, shoved it 17 mm, and the jaws closed on air — while the
  graze on the way past still read as "both pads in contact". Derive the
  structure's extent from the working geometry (`CLAW_PENDANT_BOT =
  CLAW_JAW_Z + CLAW_PAD_HALF_H`) rather than re-typing a number, and
  pytest-guard the derivation. It broke twice, the second time because the
  pad height changed and the constant did not follow.
- **A camera on a tool is aimed by rendering, not by arithmetic.** The claw's
  eye took three placement passes: above the arm's root it saw mostly its own
  arm; outboard helped; *partway down the arm* put the arm behind it and
  halved the working distance to ~65 mm.

### The actuator function

Add a `<tool>_actuator_xml()` next to `pen_actuator_xml()` /
`claw_actuator_xml()`, and wire it into **all three** world generators
(`write_hub_world`, `write_hub_rack`, and `home/world.py`'s
`rack_and_modules_xml` consumer). A module's actuators belong to the module,
conceptually driven by its own ESP32 across the wireless data link.

---

## 4. Contact and control rules

These are the house rules, in the order they have cost the most time. All of
them are guarded by tests; none of them are negotiable per-tool.

1. **`friction` without `priority="1"` does nothing.** MuJoCo combines pair
   friction as the elementwise MAX. This has bitten the caster, the pen pads,
   the peg, and the whiteboard. If you want a *low* friction, you must claim
   priority.
2. **Held contact wants a hard `solimp`, not a solver mode.**
   `coupling.GRIP_SOLIMP = "0.99 0.999 0.0001"` on the claw's jaw pads took
   slip over a 100 mm lift from **−21.7 mm to −0.13 mm**. The obvious cure was
   MuJoCo's `noslip` post-solve pass; it cost 2.8× the step time globally and
   had to be toggled per phase. Fix drift at the specific contact that drifts.
3. **A released round body rolls forever, and sliding friction will not stop
   it.** ⓘ *Found by building the dispenser.* MuJoCo's default `condim=3`
   solves no rolling resistance at all, so a dropped sphere keeps going.
   Measured on a seed released with 0.15 m/s of residual motion:

   | contact model | travelled in 6 s |
   |---|---|
   | `condim=3`, friction 0.7 | 586 mm, still doing 97 mm/s |
   | `condim=3`, friction 1.0 `priority=1` | **586 mm** — identical to the mm |
   | `condim=6`, friction `0.9 0.02 0.005` | **14 mm**, stopped |

   Cranking μ changed nothing because **a rolling ball is not sliding**.
   Rolling resistance is a different friction dimension and needs
   `condim="6"` before it exists. Sibling of rules 1 and 4: the fix goes on
   the geom that needs it, with `priority`, in the *right dimension*.
4. **`noslip_iterations` is 0, always and everywhere** (issue #3). Always-on
   noslip half-seats the jittered coupling — the peg seats by *sliding*, and
   noslip suppresses sliding — and a runtime toggle is global state that leaks
   across fixtures and, in the shared PluggyWorld model, across robots.
5. **Where a *joint* creeps rather than a contact, use `frictionloss`.** A
   velocity servo commanded 0 resists speed, not force, so the parked base
   walked under tool load until the wheel joints got the parking brake a real
   gearbox provides for free (the plotter's square: 63 % → 99 % inked).
   ⚠ The brake creates a **stiction deadband**: wheel commands under
   `frictionloss/kv` (0.1 rad/s) do not move a stopped wheel at all, so
   P-turn controllers must go through `control.turn_command`.
6. **Every position setpoint is RAMPED, never written across a gap.** A stiff
   servo handed a step delivers the whole difference as an impulse. It has
   thrown a module clean off the fork twice (an 80 mm carriage jump, a 184 mm
   lift jump — both poles open, module yawed ~100°) and batted a gripped block
   out of the jaws. Use `control.slew` for wheels, `PenPlotter.ramp` for
   actuators generally, `ClawTool.set_lift` / `.jaws` for those axes. Every
   new tool axis needs its own speed ceiling — a real screw or servo has one.
7. **Judge on a physical criterion, never on a command.** A commanded grip is
   a belief; contact is a fact. The pattern, repeated per tool:
   `module_power_contact` (electrical continuity), `ClawTool.holding` (both
   pads on the object), `pen_on_board` (ink is a contact). `on_fork` is a
   *position heuristic* with 3–4 cm tolerances — it reported True right
   through a module being ejected from the fork.
8. **Calibrate by measurement, never from the model.** Nominal geometry never
   contains droop or lean: the pen tip hangs **72 mm** below the peg against a
   47 mm estimate, and the claw's grip point was 12.5 mm off its nominal
   along-track offset. On hardware this is a one-off jig measurement, not a
   sensor.
9. **Converge a height; do not correct it once.** The grip follows the lift
   command at only ~0.87:1, because droop and lean both change as the arm
   descends. One correction left the pads 11.6 mm high — a grasp that *looked*
   right, held the block by its top 12 mm, and lost it the moment the lift
   took the weight.
10. **Stowed is the driving configuration.** Retract the arm before any drive.
   The rack taught this when an extended fork swept a module off its trays.
11. **Never teleport a robot that is carrying a tool.** The module is a free
    body held only by gravity; writing the base's `qpos` leaves it behind in
    mid-air. The giveaway, the day it happened, was a calibration reading
    `dz/dlift = 0.000` — the pen was lying on the floor.
12. **Driving to a DESTINATION is not driving through a WAYPOINT.** ⓘ *Found
    by building the dispenser.* `drive_toward`'s default pure-pursuit law is
    built for sweeping through a chain of path waypoints; aimed at a final
    pose closer than its own overshoot, it settles into a stable orbit
    *around* the target — 900° of turning to cover 200 mm. Pass
    `slow_radius=` for any terminal approach and it pivots-then-glides
    instead. Long approaches hide this completely (the claw stages 0.45 m
    back, the plotter drives ~1 m; both measure ~185°), so a tool with
    **short hops between work points is the case to check**.
13. **A P-controller that stops commanding at the target coasts past it**,
    because `slew` rate-limits the wheel command. Settle and re-check. On the
    approach to the board this was a 17° overshoot the caller could not see,
    and squareness matters: a yaw error θ swings the pen's depth by
    `110·sin θ` across the carriage's sweep — 14 mm at 7.5°, most of the
    quill's whole travel, which is why one edge of a figure came out blank.

---

## 5. The build sequence

This is the order that worked twice. Each stage exists to make the next one's
failures legible; skipping ahead has never once saved time.

### 0. Measure the tool's demands against the envelope — before drawing it

Write a headless probe against `hub_world.xml` with an existing module on the
fork, and ask the questions from §2 about *your* tool: how much moment does
its shape spend? How much force does it need to exert? Does it need an axis
the robot already has?

ⓘ *Found by building the dispenser* — and it belongs second on that list:
**what tolerance class does the job need?** Millimetres, or centimetres?

This is not a detail, it is a fork in the whole controller. The claw needs
millimetres, so it carries a staging point, a lateral P-controller with a
runway, and up to three back-up-and-take-another-run-at-it retries — all to
squeeze a 26 mm block into a 62 × 28 mm capture window. The dispenser needs
centimetres, because a sown seed lands where it lands, so it drives straight
in once and reports its residual honestly. Copying the claw's approach into a
centimetre-class tool would have been ~150 lines of machinery for nothing —
and worse, it would have meant touching the swap's travel constants, which
implicitly contain mm-scale wheel slip that anything else must re-verify.

Decide the class before you write the controller, and say so in its
docstring. Ask the companion question too — **how far apart are the work
points?** A tool that hops 20 cm between them needs `drive_toward`'s terminal
mode (rule 12); one that drives half a metre or more between them does not,
and that difference is exactly why the orbit stayed hidden until the fifth
tool.

Not everything needs measuring at this stage: the dispenser's whole stage 0
came out on paper (magazine on the peg axis → zero moment; a tool that
releases rather than presses → no lean-pad demand; ~58 g of structure on the
120 g budget → in class), and the mechanism then worked first run. Arithmetic
is fine when it is arithmetic *against measured constants* — it was the
pre-measurement guesses that were wrong both earlier times.

Both tools were redesigned by this stage. The claw stopped being an angled
arm and became a pendant; the drawing tool acquired a lean-pad on the *robot*
because the measurement said a gravity latch cannot push. The pre-measurement
arithmetic was wrong both times (it predicted ~25 g at 150 mm; the truth was
200 g), which is exactly the argument for measuring.

### 1. A tolerance spike, if the tool proposes a new interface

`scripts/hub_spike.py` (the coupling) and `scripts/schuko_spike.py` (docking)
are the template: a standalone rig with **no robot**, an ideal compliant
carrier, and a scripted sweep across misalignment. `scripts/noslip_spike.py`
is the variant for a *policy* question rather than a geometry one.

You need this only if your tool adds a new mating surface (a socket, a dock, a
fixture it must engage). A tool that only hangs on the existing peg inherits
the envelope already measured.

### 2. Module geometry in `hub/coupling.py`

Add the face and the actuator function (§3), then regenerate:

```
uv run python -m pluggybot.hub.coupling      # hub_world.xml + hub_rack.xml
uv run python -m pluggybot.home.world        # home_world.xml + meta
```

Both committed pairs are tested against their generators, so a stale file
fails the suite rather than silently drifting.

### 3. The controller class, in `src/pluggybot/hub/<tool>.py`

`PenPlotter` and `ClawTool` are the two worked examples, and they share a
shape: hold `(model, data, swap)`, resolve actuator/site ids in `__init__`,
then

- **state readers** — where is the working point (`pen_world`, `grip_world`),
  and the physical criterion (`pen_on_board`, `holding`);
- **primitives** — ramped setpoint moves, one per axis (`ramp`, `set_lift`,
  `jaws`), plus a `_face(heading)` that settles and re-checks;
- **calibration** — a measured mapping from commands to world coordinates;
- **the verbs** — the two or three things the tool actually does
  (`draw`; `pick_up` / `set_down`), each returning a dict of *measured*
  outcomes rather than a bare bool.

Roughly 300–500 lines, most of it comments recording what the numbers cost.

### 4. A demo script with a filmstrip

`scripts/draw.py` and `scripts/pickup.py`. The demo is not decoration — it is
how geometry bugs get found. Three of the coupling's original bugs were found
by filmstrip and phase telemetry, **none by reasoning**; "it's standing on its
tail" was invisible in scalars.

Conventions: `--view` for the live viewer, headless-by-default with a saved
PNG (filmstrip + result panel), and honest reporting — `scripts/home_draw.py`
reports its failing stow as a failure rather than claiming success.

⚠ **Pick camera angles by sweeping azimuth at the moment of contact.** The
pen filmstrip's az=150 sits *behind* the board, so the whole drawing phase
renders as a grey rectangle. Reasoning about the geometry got this wrong —
and it got the dispenser's first filmstrip wrong too, in a demo written by
someone who had just written this warning. Sweep. It costs one probe.

ⓘ *Also found by building the dispenser:* **for a small working point, a
tracking camera on the module body is the wrong instrument.** Tracking
`module_seed` put the magazine behind the chassis and a 16 mm seed was
invisible at all eight azimuths tried. Aim a **free** camera at the tool's
working *site* instead — outlet, pen tip, grip point — raised a little and
pulled back enough to keep the robot in frame for context.

### 5. Pytest regressions, each shown failing first

`tests/test_drawing.py`, `tests/test_gripper.py`, `tests/test_hub_coupling.py`.
The rule from CLAUDE.md applies without exception: **every debugged failure
becomes an assertion, and the assertion must be shown to fail without the fix
— a regression test that cannot fail is décor.**

What the existing tool tests assert, as a checklist for yours:

- the tool can be **fetched** from its bay and is **electrically powered**
  after the pick (`module_power_contact`);
- the working point lands where it should, judged **per axis** — a single
  radial tolerance hid that the claw's jaws capture a 62 mm lateral gap but
  only 28 mm fore-aft;
- **the job itself succeeds**, on the physical criterion (block off the floor;
  ink on the board);
- **the module is still seated afterwards** — using the tool must not unseat
  it;
- **derived constants stay derived** (`test_pendant_stops_at_the_jaw_tops`);
- **design constraints that must not be quietly forgotten** get an assertion
  too (`test_the_robot_cannot_see_what_it_grasps`).

ⓘ *Found by building the dispenser:* **do not share a fixture that holds a
consumable.** The dispenser's landing test used a module-scoped "tool picked
and calibrated" fixture, passed the whole file, and failed when run alone — it
had only ever been seeing seeds an *earlier* test had dispensed. Function
scope costs ~5 s a test here and is the honest price; a fixture that carries
mutated state between tests is a shared world pretending to be a fresh one.
Run at least one of your new tests in isolation before believing the file.

And pin the envelope *from both sides*: `test_yaw_4deg_is_outside_the_envelope`
asserts a known limitation, with a comment saying that if it starts passing,
that is good news wearing a test failure — re-measure and update SimNotes,
don't just delete the assert.

### 6. Write it down, and re-emit the fixtures

- a SimNotes section: what was measured, what broke, what is still open;
- PluggyPlan status if it closes or opens a milestone item;
- a CLAUDE.md command entry for the demo script;
- **fold this doc's own gaps back in** — that is what makes the next tool
  cheaper than yours was, and it is half the point of the exercise;
- **regenerate the protocol fixtures** — new geoms change the scene:

```
uv run python -m pluggybot.telemetry.scene                      # room_hub
uv run python -m pluggybot.telemetry.scene models/home_world.xml
MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py \
  --record protocol/telemetry.hub_lifecycle.jsonl.gz
MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --world home \
  --record protocol/telemetry.home_lifecycle.jsonl.gz
```

ⓘ *Found by building the dispenser:* **adding a tool is a telemetry event,
and there is a census test that will tell you so.** A new module adds dynamic
bodies — the module, any moving parts, any payload — and every one of them
costs a pose in every keyframe. `tests/test_room_hub_coverage` pins that count
deliberately (16 → 21 for the dispenser: module + shuttle + three seeds), so
update it consciously rather than reflexively. No `protocolVersion` bump is
involved — the artifact's *content* changed, not its *shape* — but the website
repo must still re-vendor `protocol/`.

There is one scene *and* one recording **per world**, and the fixture test
fails when they are stale. If your tool wants a new **visual hint**, note that
`telemetry.protocol.VISUAL_HINTS` is a two-repo contract: *adding* a hint is
additive (the website falls back to raw primitives), *renaming* one is a
breaking change in both repos. And never encode a hint as a geom colour — the
robot's own cameras render rgba.

---

## 6. Rack integration

### Bays

`HUB_STATION_YS = (0.125, -0.125, 0.375, 0.625)` — four tool bays at 0.25 m
pitch, plus a charge bay at `CHARGE_BAY_Y = -0.375` continuing the same pitch.

**The tuple is APPENDED to, never inserted into.** Bay↔tag pairing is *by
index* (`bay_tag_id` finds the nearest station and indexes `BAY_TAG_IDS`), and
every demo and test names its bay as `HUB_STATION_YS[0]`, `[3]`, and so on.
Reordering for tidiness would silently re-point all of them. Bay C/D are
already out of y-order for exactly this reason.

Geom names come from `bay_prefix(i)`: `baya_`, `bayb_`, `bayc_`, `bayd_`.

### Tags

Real tag36h11 AprilTags, generated by `hub/tags.py` and rendered as cube
textures on flat plates:

| marker | id(s) | physical size |
|---|---|---|
| rack | 0 | 120 mm — read from across the room |
| charge bay | 3 | 30 mm |
| bays A–D | 1, 2, 4, 5 | 30 mm — read from arm's length |
| modules | 10 (lcd), 11 (plug), 12 (pen), 13 (claw) | 30 mm |

A new tool needs a new entry in `MODULE_TAG_IDS` (next free: **14**), and a
new *bay* needs one appended to `BAY_TAG_IDS`. Tag size is a range decision,
not a detail — a tag36h11 must span ~25–30 px to decode, so the physical size
sets the distance at which it can be seen. Ids are **not renumbered**: the
gap at 3 exists because renumbering would invalidate every generated PNG and
model for cosmetics.

Two gotchas already paid for: textures must be `type="cube"` (a `2d` texture
maps through geom texcoords, which primitives do not carry — the plate renders
flat grey and nothing ever decodes), and the PNG upscale must be
nearest-neighbour, because a marker is data and interpolation softens the cell
edges.

### Bay count is a real limit

The rail is `2 × RACK_HALF_W` = **1.36 m** and currently carries five stations
spanning y_local −0.375 … 0.625. **A sixth station at the same pitch (y =
0.875) falls outside the side post at 0.68**, so a fifth *tool* requires:

1. growing `RACK_HALF_W` to ≥ ~0.93 (a 1.86 m rail), **or** a second rack;
2. appending a bay tag id;
3. re-checking the rack against the room it stands in. Today it sits at world
   (−0.90, 5.99) yaw −90° in `room_hub` and spans x = −1.58 … −0.22 against a
   west wall at −2.0. A 1.86 m rail would span −1.83 … +0.03 — worth checking
   before assuming.

The rack is a **free body**, deliberately: whether the robot's presses scoot
or tip it is a question the sim should measure, not assume. (It has an answer:
without the wall braces it scooted 9.6 mm under a sustained charge-bay press.)

### Approach and ranging

`hub/mission.py` handles the navigation half. Two things a new tool inherits
rather than reimplements:

- **Range off the bay's own marker, not the rack's.** The rack tag hangs at
  the rack's centre and is outside the dock camera's view from the outer bays,
  so bays C/D silently fell back to odometry — whose 13–21 mm drift a *pick*
  forgives (the V self-centres during the lift) and a *return* does not.
  `TagSpotter.bay_range` fixed it to ≤ 2 mm.
- **Steer on one named marker.** Steering on "whatever tag is visible" once
  locked onto the charge bay's marker and dragged a module 22 cm toward the
  wrong bay.

---

## 7. Known gaps a new tool inherits

Do not rediscover these; they are open, and they are not yours to fix unless
your tool makes them worse.

1. ~~**The pen does not stow.**~~ **CLOSED by issue #10** — and the rule this
   entry drew from it was pointing at the wrong axis, so read the correction
   rather than the original. The controlled experiment (same bare world, same
   script, same default `put_back`, five modules) was right:

   | module | hung after `put_back` | rack-frame x |
   |---|---|---|
   | lcd / plug / claw / **seed** | ✔ | 0.090–0.096 |
   | **pen** (before the fix) | **✘ — still on the fork** | **0.444** |

   The conclusion drawn from it — "the pen's assembly stands ~26 mm proud of
   its plate in **x**, and that is what fouls" — was wrong. Standing proud in
   x is harmless; the rack leaves that air empty. What fouled was **z**: the
   rail sat in the same height band as the bay's tray brackets, so the
   module could not be raised the 31 mm a set-down needs without jamming
   under them. See SimNotes, "The pen would not stow".

   What survives, and is now *measured* rather than suspected, is this
   entry's own last sentence: **keep your tool's parts below `-TOOL_HALF_Z`
   and you inherit none of this.** The dispenser obeyed it and stows at
   2.8 mm of bay error. The pen broke it in two places — the rail (fixed by
   moving it below the plate, where the rule says it belongs) and the
   carriage block, which *cannot* go there because it has to reach the quill
   at the pen line. That leftover is why tools with a moving axis need a
   stow pose (§2, "What a carried module does in transit").
2. **Nothing can autonomously find a floor object.** The LIDAR plane is
   223 mm up and the nav camera is blind to the floor inside 0.48 m, while a
   grip point sits ~285 mm ahead of the axle. Grasps run open-loop from a
   memorised pose. Marked delivery zones are the honest workaround.
3. **Yaw at the coupling has ~2° of margin** against navigation's ~0.5°
   settle. Known v2 levers: y-chamfered trays, a squaring press against the
   rack's back wall, softer yaw compliance.
4. **The plotter's carriage loses ~19 % of its commanded travel while
   sweeping with the pen down** (~0.81:1 against ~1:1 at rest). The loss is
   kinetic — drag reverses with direction and yaws the module — so a
   calibration that settles before each reading cannot see it by
   construction. Characterised, not fixed. Any tool that drags against a
   surface should expect the same.

---

## 8. Checklist

```
[ ] which axis does it BRING, which does it BORROW?
[ ] moment / force / mass demands against §2, before drawing anything
[ ] tolerance class: millimetre or centimetre? (decides the whole controller)
[ ] payload, if any: retained by GRIP or by GEOMETRY?
[ ] parts below -TOOL_HALF_Z where you can -- height, not x, is what jams a
    stow: z -30..-9 mm of the module frame belongs to the bay's brackets
[ ] a moving axis? decide its STOW POSE and return to it before stowing
[ ] tolerance spike, only if it adds a new mating interface
[ ] a free bay: five exist; a sixth means growing the rail AND re-checking
    the rack against both rooms
[ ] face + actuator fn in hub/coupling.py; identity tag on the +x face;
    new ids in tags.py (MODULE_TAG_IDS, BAY_TAG_IDS if a bay was added)
[ ] regenerate: hub.coupling, home.world
[ ] controller class in src/pluggybot/hub/<tool>.py, tolerance class stated
[ ] demo script: filmstrip (free camera on the working SITE, angle swept),
    --view, honest verdicts
[ ] pytest: fetch, power, aim (per axis), the job on a physical criterion,
    still-seated-after, derived constants; no shared consumable fixtures
[ ] every new assertion shown failing without its fix
[ ] run at least one new test in isolation, not just the whole file
[ ] SimNotes section; PluggyPlan status; CLAUDE.md command entry
[ ] fold this doc's gaps back in -- mark them so the next builder sees them
[ ] update the dynamic-body census in tests/test_telemetry.py
[ ] regenerate protocol scene + recording fixtures (one of each per world)
[ ] MUJOCO_GL=egl uv run pytest -q; uv run ruff check src/ scripts/ tests/
```
