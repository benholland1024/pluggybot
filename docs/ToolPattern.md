# Tool Pattern — how to build the next module

The recipe for adding a tool to the quadruped's rack (rewritten for the arm
in #407). A tool hangs on the rack beside the dock (`legs/rack.py`), the
robot takes it on the fork at the tip of its two-joint arm (`legs/arm.py`,
`legs/swap.py`), and it carries the tool over its nose while it walks.
Three hand-built tools hang in the rack's own bays — the LCD (bay A), the
pen (B) and the claw (C) — and a tool the robot builds hangs on the
built-tool rail, three more bays along the same board. What carried over
from the wheeled rover (`rover-final`)
is the coupling — a split peg, gravity as the latch, the peg's two conductors
as the power — and the build discipline; its lift, rack, lean-pad force
table and seed dispenser are history.

The next builder may not be a person: the workshop's validator
(`workshop/validate.py`) reads §2's numbers from `legs/arm.py` and
`legs/rack.py` and refuses a spec outside them, so §2 stays numbers, not
stories.

Read this alongside, not instead of:
- `docs/SimNotes.md` — how each number was found: "The quadruped's arm, its
  coupling and the rack", "The rack at the arm's reach, and the swap",
  "Drawing on legs", "The claw on legs", "The workshop on legs". This doc
  says what the constraint is; SimNotes says how we found out.
- `docs/Parts.md` — the hardware the sim parameters model.
- `CLAUDE.md` — the house rules in their short form.

---

## 1. What a tool module is

**A tool is not cargo. It is a kinematic extension the robot acquires by
picking it up.** The robot already owns some axes, and the module supplies
only the ones it lacks:

| axis | who owns it | notes |
|---|---|---|
| x, y, yaw | the legs (the walking policy) | it walks no slower than ~0.2 m/s and keeps turning after a stop: it places the body to centimetres, never millimetres |
| reach and height in the arm's plane | the arm's two pitch joints, the plate held level by a parallelogram | lying, the floor 0.35–0.82 m ahead; a board's 0.19–0.41 m from 0.43 m out; a bay's peg at 0.50 m |
| a stance | the posture: lying or standing | tools work LYING — standing, the policy never quite stops |
| **anything else** | **the module** | sideways millimetres are the module's: the pen's and the claw's L12-100 slides |

The pen brings sideways travel (a carriage) and a sprung quill, and borrows
the arm for the board's height and the press. The claw brings sideways travel
and a grip, and borrows the arm for the floor. The LCD brings a screen and
nothing that moves.

Before designing anything, answer: **which axis does this tool bring, and
which does it borrow?**

The module also gets, for free: **power** through the peg (§3), and **data**
wirelessly (its own ESP32). It has **no tag of its own**: a tool is known by
its bay — the bay's pair of tags and its presence switch
(`coupling.bay_switches`).

---

## 2. The coupling envelope

The mechanical contract. A new tool fits inside it; it does not renegotiate
it.

### The latch

A tool hangs by a **220 mm split peg** (`legs.rack.PEG_HALF` 0.110: the
rover's 75 mm plus the fork's sideways capture) resting in two upward-open
V-trays at ±45 mm (`TRAY_Y`). The fork takes it outboard, at ±85 mm, in V's
of 60° with 53° end-ramps faced slippery (`legs.arm.ForkSpec`). The verbs
are **in under the peg, up 56 mm (`arm.LIFT`), back out** — and the reverse
to hang it back. **Gravity is the latch**: no spring, magnet, catch or lock.

The V's are 60° (the rover's were 45°) because of stairs: the parallelogram
holds the plate at the torso's angle, and coming down a flight the torso
pitches 33° nose-down, which laid a 45° V's flank under the peg's friction
angle.

### Capture

| axis | the fork takes | notes |
|---|---|---|
| across (along the peg) | −20 … +25 mm | ±18 reliable; the line-up gate is `rack.LINEUP_ACROSS` 15 mm |
| yaw | 6° | at 8° the far V is past its mouth; the gate is 4° |
| along (the aim) | −10 … +10 mm | measured off the bay's tags to under 1 mm |
| height | ±10 mm | |

The approach delivers this (§6): 41 of 41 starts up to ±0.3 m and ±30° off
took the tool and hung it back.

### The tool envelope — what a validator reads

| rule | limit | where |
|---|---|---|
| mass, plate and peg included | **0.40 kg** | `arm.TOOL_MAX_KG` |
| centre of mass | **on the peg's line or ahead of it** (away from the robot), at most **60 mm**, at the stow pose and at every axis's two ends | `arm.TOOL_MAX_AHEAD_M` |
| moment about the peg | **0.35 N·m** | `arm.TOOL_MAX_MOMENT_NM` |
| hangs plumb | within **2°** at its stow pose: hung, nothing holds it level | `rack.HUNG_TILT_DEG` |
| hangs centred | its centre of mass within **25 mm** of the peg's middle at its stow pose: the trays hold the peg at ±45 mm (measured: 35 mm did not hang back with the fork 15 mm toward it) | `rack.HUNG_SIDE_M` |
| drop | nothing more than **0.20 m** under the peg at any pose: carried, it clears the LIDAR's plane by 2 cm | `arm.TOOL_MAX_DROP_M` |
| behind | nothing behind the plate's back face (the fork's prongs, bridge and lean-pad) | `validate` |
| the peg's ends | nothing where the fork's V's or the trays hold them | `validate.FORK_ZONE`, `TRAY_ZONE` |
| hung | nothing within 10 mm of the rack's board; nothing in front of its bay's tags, the plate's own slab included (the camera looks past the plate) | `validate.BOARD_CLEAR_M`, `TAG_ZONE` |
| the rail | nothing in reach of the rail the trays hang from, 0.11–0.13 m over the peg, hung or lifted by a pick's 56 mm | `validate.RAIL_ZONE` |
| power | the parts' draw plus the module's ESP32 under **12 W** | `coupling.PEG_POWER_W` |

Measured (`scripts/arm_spike.py --envelope`, `--retention --stairs`): a
0.60 kg tool stayed seated through a 1.0 m/s trot and a stop with its centre
of mass up to 60 mm off its peg; down #388's flight (ten 0.18 m risers) at
60 mm ahead, none of 18 lost. The arm holds 0.61 kg — the claw and a 0.4 kg cube — straight
out at 39 % of its motors' continuous rating, so the 0.40 kg ceiling leaves a
payload room.

**Ahead, not behind.** A centre of mass ahead of the peg leans the tool 6°
onto the lean-pad, a round bar standing 3 mm behind the plate; behind the peg
and below it is where the pad's post is. The pad also takes a pressing
tool's reaction after 4° of swing.

**What a tool may push** (`scripts/mechanism_spike.py --window`: a push at
the claw's jaws, 175 mm under the peg; SimNotes, "Opening a box from
lying"). A tool hangs on its peg as a pendulum, so it holds a push one or two
ways only:
- **down:** at least 6 N, the ramp's end;
- **toward the robot:** 3 N, leaning on the pad;
- **away from the robot:** nothing — it swings 50° at 1 N;
- **up:** about 0.9 N. Pushed up under its peg it is an inverted pendulum,
  beaten at about its weight × (its centre of mass under the peg ÷ the
  push's depth under the peg). It tips onto the pad and rolls out of its V's.
  Its whole weight is needed only for a lift through its centre of mass;
- **sideways:** about 1 N rolls it off one V.

A tool that grips something which holds it is not free to swing, and what
must hold then is the grip's pull.

### What a carried tool does

- **It swings.** Up to 22° off plumb on a stop from 1.0 m/s: a carried tool
  needs clear air 57° either way, and carry clearance is sized by the swing.
- **A pivot unseats it.** At the drive's full 1.0 rad/s the tool swung out of
  its V's for 160 ms against the peg's 200 ms holding capacitor, so a body
  carrying turns at most `legs.body.W_CARRY` (0.45 rad/s).
- **Stairs are the open risk** (§7): the house has no flight (#280 closed).
- **A fall throws it**, whatever holds it; the arm folds as the torso passes
  60° (`arm.FOLD_ON_FALL_COS`).
- **It is the body's own to its senses** (`QuadMission.carry`): out of the
  map, the depth cloud, the press and a stairs policy's height scan — ⚠ a
  tool's geoms in group 0 read as terrain to that scan.

### A moving axis needs a stow pose

An axis left wherever the last job ended is not a pose anyone designed. A
built tool's spec carries `stow` per axis: the validator checks every rule
at the stow pose and at each axis's two ends, the module compiles AT its
stow (each joint's `ref`, so `qpos0` is the stow) with its servo holding
there, and every hang-back first returns each axis of any tool to its rest
(`steps.carry_configuration_routine`). Built at 0, a flap whose stow is −90°
hung 14° off plumb and was lost from the moment it hung. A claw hung back
holding a cube is a cube hung on the rack, so a stow sets the cube down
first, or lets it go where it stands.

### If the tool carries a payload

- **Retained by grip** (the claw's cube) — the hold is a force and can be
  lost. MuJoCo's soft contact lets a held body creep at about
  (1 − d)/d · g/b, so the pads are stiff (`CLAW_PAD_SOLREF` 0.006,
  `CLAW_PAD_SOLIMP` 0.99/0.999 with a 10 µm width) and the jaw servo
  saturates at its stall force (`CLAW_GRIP_KP` 2000, 10 N): a held cube
  creeps 0.03–0.19 mm/s from 60 to 320 g (SimNotes, "The claw on legs"),
  and the jaws hold 0.40 kg, never reading unheld; 0.50 kg shaken across
  them does (`challenge/bench.py`'s `MAX_KG`).
- **Retained by geometry** (the rover's seed magazine) — no pose in which the
  payload has anywhere to go: nothing to tune. Prefer it whenever the payload
  need not be grasped.

A payload counts against the moment like structure does, at whatever offset
it sits.

---

## 3. Module anatomy

Everything below is in `src/pluggybot/legs/rack.py`.

### Generated for you

`tool_xml(name, pos, yaw, mass, face)` emits a free body whose **peg's axis**
is at `pos`, its +x face turned toward the robot: the plate (20 × 40 × 60 mm,
`coupling.TOOL_HALF_*`) and the peg (`peg_xml`). `tool_default` is its geoms'
class. The plate and peg weigh `MODULE_MASS` (0.129 kg); a tool's face adds
to that (the LCD 0.151 kg in all, the pen 0.190, the claw 0.239).

### The peg is also the connector

`peg_xml` splits the rod into **two conductors round an insulated centre**,
and the fork's left and right V pairs are the two poles: the seat's preload
is the tool's weight, and the seating slide wipes the contacts. Two things
follow for every tool:

- **`PEG_FRICTION` 0.4 with `priority="1"`.** MuJoCo's default μ = 1.0 sits
  the peg on its sliding threshold in a V, and without `priority` a low
  friction does nothing (pair friction combines as the elementwise MAX).
- **`tool_power(model, data, name)` is the seating check**, pole by pole: a
  tool on the fork with one pole open is a real failure of a two-point latch.
  `on_bay` is the hung check: the peg down on both flanks of both trays, the
  tool plumb.

### The `face`: the tool's own parts

A face string in the module's frame: **+x toward the robot that carries it,
−x toward the rack's board when it hangs**, z up, the peg's axis along y at
z = 22 mm (`PEG_ABOVE_BODY`). `tool_face(name)` returns it; `TOOL_KG` and
`FACE_KG` carry the masses.

| element | pen | claw | LCD |
|---|---|---|---|
| structure | rail | rail, crossbar, jaws 175 mm under the peg | screen |
| joints | carriage, quill | slide, two jaws | — |
| actuators | carriage (the quill is a passive spring) | slide, jaws (one servo through a rack and pinion, two position actuators commanded as one) | — |
| working site | `pen_tip` | `claw_grip` | — |
| payload | — | grip-retained | — |

Rules from the builds, each paid for:

- **Joints ≠ actuators.** The pen's quill is a sprung slide with no motor
  (`PEN_QUILL_STIFFNESS` 60 N/m over 20 mm): pen pressure becomes a design
  constant rather than a positioning problem. Compliance where the tool meets
  the world is cheaper than accuracy in the arm.
- **Model the actuator you would buy.** The carriages are `kp` 2000 because a
  lead screw does not yield to drag (at a hobby servo's gain the rover's
  figure came out 12 mm off); the claw's jaws are an FS90MG at its stall
  force. Part choices, not tuning knobs. A lead screw holds unpowered, so a
  position servo parked at its target is the honest model of a racked tool.
- **Mount in front of the plate, not inside it.** MuJoCo filters only
  parent-child contacts: a grandchild part collides with the plate.
- **The structure must not occupy the working space** — and **a hung tool
  must not hide its bay's tags** (the validator's `tags` rule; the claw's
  crossbar is SimNotes, "The claw on legs").
- **A camera is aimed by rendering, not arithmetic** — and may not be the
  tool's: the claw finds a cube through the D435's colour imager on the body
  (SimNotes, "The claw on legs").

### The actuators

`tool_actuators_xml(tools)` emits the actuators of the tools a world has; add
the new tool's there. A module's actuators are driven by its own ESP32 over
the wireless link, and `data.ctrl` is written only by the modules in
`tests/test_procedure.py::CTRL_WRITERS`.

---

## 4. Contact and control rules

All guarded by tests; none negotiable per tool.

1. **`friction` without `priority="1"` does nothing** — pair friction is the
   elementwise MAX. A low friction must claim priority.
2. **A held contact wants a hard `solimp` on that contact**, not a solver
   mode (the claw's pads, §2).
3. **A released round body rolls forever under `condim=3`**: rolling
   resistance needs `condim="6"` (the rover's seed rolled 586 mm; 14 mm at
   condim 6).
4. **`noslip_iterations` is 0, always** (issue #3): noslip half-seats the
   coupling, which seats by sliding.
5. **Every setpoint is RAMPED** — the fork moves at `arm.FORK_V`, the
   carriages at their speed. A stiff servo handed a step delivers an impulse:
   it has thrown a tool off the fork and batted a block out of the jaws.
6. **Judge on a physical criterion, never on a command**: the poles
   (`tool_power`), the hang (`on_bay`), the hold (both pads within 0.5 mm of
   one body, by `mj_geomDistance` — stiff pads' contacts flicker step to
   step), the ink (a contact). A fork move fails only when it is out of
   reach; what happened is the world's.
7. **Calibrate by measurement**: the pen finds the board's face by touch,
   the claw measures where a held cube hangs before it puts it down.
8. **The arm travels folded**: to its stow empty, or to the carry pose with a
   tool aboard, the shoulder first (`steps.travel_pose`). An extended arm on
   a walk is an arm that sweeps.
9. **A stopped robot keeps turning** — the flat policy holds no heading. It
   stands `rack.SETTLE_AFTER_WALK_S` (3 s) and measures again, and a walk-in
   stops `rack.SETTLE_DRIFT` (1.9°) clockwise so the settle brings it
   square. After any big arm move it measures again too: swinging a tool up
   and down moved the torso 25 mm.
10. **Never teleport a robot carrying a tool**: the tool is a free body held
    by gravity and stays behind in mid-air.
11. **Every terminal loop has a budget and an explicit answer** — a walk-in
    has three tries, a search its patience, a swap `TO_BAY_S`.

---

## 5. The build sequence

Each stage makes the next one's failures legible.

### 0. Measure the tool's demands against the envelope, before drawing it

Mass, the centre of mass's lever, the drop, the draw: arithmetic against
§2's measured constants is fine, and `arm_spike.py --envelope` and
`--retention` fly a tool's mass and lever when arithmetic is not enough. Ask
too: **what tolerance class does the job need?** The claw needs millimetres,
so it lies down at a cube steered by the cube's own tag and tries three
times; a centimetre-class tool walks up once and reports its residual.
Decide before writing the controller, and say so in its docstring.

### 1. A tolerance spike, only for a new mating surface

`arm_spike.py --capture` is the template: the coupling placed at a bay and
swept across misalignment, no walking. A tool that only hangs on the peg
inherits the envelope already measured.

### 2. The module in `legs/rack.py`

A face, its masses, its actuators, and a bay in `TOOL_BAYS`; then regenerate
the scenes (`uv run python -m pluggybot.telemetry.scene [--pair]`) and the
parts list (`uv run python -m pluggybot.rack.catalog`).

### 3. The routines and the verbs

The body's routines are mixins of `QuadMission` — `legs/draw.py` (the pen),
`legs/claw.py` (the claw), `legs/survey.py` (the LCD's census),
`legs/swap.py` (fetch and stow) — each returning a record of what it
MEASURED. A program reaches them through verbs in `procedure/steps.py`, and a
tool's axes and sensors through the registries in `procedure/axes.py`; the
language does not change.

### 4. A spike with a filmstrip

How geometry bugs are found: `arm_spike.py --view`, `scripts/solve.py` for a
job's ladder A. Pick camera angles by sweeping azimuth at the moment of
contact, and aim a free camera at the working site rather than tracking the
module body.

### 5. Pytest regressions, each shown failing first

Pin the rule as cheaply as it can fail for the right reason (CLAUDE.md, the
test budget): a direct call, a stub, a bare-world rig; a flight only for a
claim about physics, and behind `--endurance` only with Ben's approval. What
the tool tests assert, as a checklist: it is fetched and powered; its working
point lands per axis; the job succeeds on a physical criterion; it is still
seated afterwards; it is hung back; derived constants stay derived. Do not
share a fixture that holds a consumable.

### 6. Write it down, and re-emit the fixtures

A SimNotes section; a CLAUDE.md line for any script; this doc's gaps folded
back in; the protocol fixtures regenerated (`protocol/README.md`). **Adding a
tool is a telemetry event**: each moving part is a dynamic body with a pose
in every keyframe, and `tests/test_telemetry.py` pins the count (39 in
`home_quad`). The website re-vendors `protocol/`; a new visual hint is a two-repo
contract and never a geom colour.

---

## 6. Rack integration

### Bays

The rack (`legs.rack.DEFAULT`) stands on the living room's south wall beside
the dock, its pegs 0.50 m up: three bays at −0.30, 0 and +0.30 m in its own
frame — the LCD at A (the east one), the pen at B, the claw at C
(`TOOL_BAYS`). **These three are permanent** (issue #277): every offered job
is written against them. The **built-tool rail** (`legs.rack.BUILT`, #407)
is three more bays on the same board and in the same frame, at 0.65, 0.95
and 1.25 m, running west away from the dock, for the tools the robot builds
and nothing else.

Every bay lives in one index space, `coupling.STATION_YS`: the rack is
stations 0–2, the rail 5–7 (`RackSpec.stations`, `coupling.built_bay_index`).
**It is APPENDED to, never reordered**: the inventory, the bay switches'
names (`bay<letter>_tray_...`, the rack `baya`–`bayc`, the rail
`bayf`–`bayh`) and every test name bays by index.

### Tags

Real tag36h11 AprilTags from `rack/tags.py`, cube textures on flat plates:

| marker | ids | size |
|---|---|---|
| the rover's rack, bays and modules | 0–19 | retired with it, not reused |
| the tower's blocks, the bench's cubes | 20–22, 23–24 | 20.8 mm, every face |
| the dock | 25–28 | 60 mm |
| the rack, a pair a bay at ±75 mm | 29–34 | 60 mm |
| the lab's plate signs | 35–37 | 120 mm |
| the whiteboards, a pair a board | 38–41 | 120 mm |
| the claw's and the census's areas | 42–46 | 120 mm |
| the built-tool rail, a pair a bay | 47–52 | 60 mm |

The next free id is **53**; ids are never renumbered. A bay's tags must stay
in the nose camera's view from its working pose: between the bays (±150 mm)
they sat 25° off its axis and 2.7° of yaw took one out of frame. Textures
must be `type="cube"` (a `2d` texture renders flat grey on a primitive), and
the PNG upscale nearest-neighbour (a marker is data).

### A built tool's bay

Before a point moves the workshop checks the envelope (`validate`), that
none of the module's names is the world's already (`seam.names_taken`), and
the RIG (`build.trial`): the module hung, taken, worked and hung back on a
bench rack with the fork on the bay's middle and at the line-up gate either
side, 0.25 s a try. Then `HubLifecycle.hang_tool(tool, bay)` takes a rail
index (A = 0), retires a built tool already there, attaches the module at
the bay's peg in the rack's own frame (`seam.rack_pose`, `rack.bay_peg`) and
recompiles the running world (a few milliseconds of wall time; 6–10 ms on
the served pair here). Every holder of the old model is rebound and
re-resolves its ids by name (`QuadMission.rebind`); the module is attached
LAST, after the robots, so only another built module's ids ever move. The
hang waits until every body is still (`seam_busy`). Hung, a built tool is
fetched and stowed as a hand-built one is (`fetch("module_<name>")`);
`scripts/workshop.py --served` flies the lot.

### Bay count is a real limit

A fourth rail bay is `coupling.BUILT_STATION_YS` (it sets the workshop's
cap), `BUILT.bays` and `.stations` appended and two tag ids — the board
follows its bays — and checked against the house: the walls, a door, the
planner's inflation round the working pose. A fourth hand-built tool needs
bays of its own the same way.

### Approach and ranging

The rack is **commissioned with the dock** (Ben, 2026-09-29;
`tool_rack_prior`), and each approach is measured off its tags. The robot
walks to a metre behind the bay's working pose (`rack.work_pose`, 0.45 m
out), looks (the rack's frame off every tag it sees, rack and rail
together: `fit_rack(seen, SPECS)`), walks in steering by the bay's tags
(`walk_in_twist`), settles, and measures the bay off its own section's tags
(`bay_aim`) — never off one tag's yaw, which square-on is a coin flip
between two mirrored solutions. Inside the line-up gate it takes the tool; outside it
backs out and tries again, three times. Another robot at this bay's working
pose (its reported pose within 0.55 m) holds it at the standoff (#418).

---

## 7. Known gaps a new tool inherits

Open, and not yours to fix unless your tool makes them worse.

1. **No lock.** Coming down a flight the fork falls faster than g at a
   step, and about 1 descent in 60 floats the peg onto an end-ramp and loses
   the tool; the levers are a slower descent, a carry pose nearer the
   torso's pitch axis, and a magnet in each V (Parts.md).
2. **Two carrying robots' tools can knock each other off** at the rack: 1 of
   20 at bays A and C, 0.60 m apart — outside the wait's 0.55 m, which reads
   only the other robot's reported pose.
3. **The walk-in avoids no robot**: steered by the tags, never the planner,
   it walked over a robot lying in the approach (the loop never leaves one
   there).
4. **Two stows in the pair tables let go** (the pen at B, −11.2 mm and
   −2.6°; the claw at C, +6.9 mm), each with the other robot working at the
   same time: recorded, not explained.

---

## 8. Checklist

```
[ ] which axis does it BRING, which does it BORROW?
[ ] mass / lever / drop / plumb / power against §2 before drawing anything
[ ] tolerance class: millimetre or centimetre? (decides the controller)
[ ] payload, if any: retained by GRIP or by GEOMETRY?
[ ] nothing behind the plate, nothing in front of its bay's tags or in
    reach of the rack's rail when hung; its centre of mass under the peg and
    within 25 mm of its middle
[ ] a moving axis? its STOW POSE, returned to before it is hung back
[ ] a tolerance spike only if it adds a mating surface
[ ] a bay: TOOL_BAYS (hand-built) or the rail (built); STATION_YS appended
[ ] face, masses and actuators in legs/rack.py; tag ids only for new bays
[ ] routines on QuadMission, verbs in procedure/steps.py, axes and sensors
    in procedure/axes.py
[ ] a spike with a filmstrip (free camera on the working site, swept)
[ ] pytest: fetched, powered, aim per axis, the job on a physical criterion,
    seated after, hung back; each assertion shown failing without its fix
[ ] SimNotes section; CLAUDE.md line; this doc's gaps folded back in
[ ] the dynamic-body count in tests/test_telemetry.py
[ ] regenerate the protocol scenes, recording and parts list
[ ] MUJOCO_GL=egl uv run pytest -q; uv run ruff check src/ scripts/ tests/
```
