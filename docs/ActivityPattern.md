# Activity Pattern — how to build a task state machine

The recipe for adding an *activity* to PluggyWorld: a puzzle, a mechanism, a
gardening step — anything the robot does to the world that the world then
has to remember. Companion to `docs/ToolPattern.md`, which covers the things
the robot picks up; this covers the things it acts on.

Written alongside its first consumer, the **pressure plate and gate**
(`activity/plate.py`, demo `scripts/plate.py`), so every rule here is one the
build actually paid for.

Read alongside:
- `docs/SimNotes.md` — the measurements behind the numbers quoted here.
- `docs/ToolPattern.md` — the sibling doc; several rules are shared and are
  cross-referenced rather than repeated.
- `protocol/README.md` — the wire format activities join.

---

## 1. Three layers, and MuJoCo owns one

This is the whole idea, and it is what makes gardening (or puzzles, or
mechanisms) cheap enough to be worth doing:

| layer | owns | example |
|---|---|---|
| **1. Rigid bodies and contacts** | MuJoCo | the plate's sprung joint; the wheel that presses it |
| **2. Task state machines** | *this doc* — Python | "is it pressed"; "is the gate open"; "is this plot dug" |
| **3. Browser visuals** | the website | the gate's swing, water pouring, plants growing |

The question is never *"can MuJoCo model soil, water, or growth"*. It is
**"can the robot make a convincing rigid-body gesture at a known place that a
state machine can verify"** — and layer 2 is that verification.

**This is the oldest pattern in the repo, not a new one.** Charging has always
worked this way: `rack_charge_contact` is a contact-derived criterion and the
battery filling is bookkeeping. So are `module_power_contact`,
`ClawTool.holding` and `pen_on_board`. Every one answers a question about the
world with a **fact read out of the physics** rather than a belief about what
was commanded. An activity is that pattern given a name, a memory, and a way
onto the wire.

---

## 2. Anatomy

An activity is one subclass of `activity.base.Activity` with one required
method:

```python
def sense(self, model, data) -> None:   # called every physics step
```

and one public surface, `self.flags` — a small dict of JSON-ready scalars that
goes to telemetry verbatim.

**An activity module owns its geometry too.** `activity/plate.py` emits its own
MJCF and holds its own state machine, exactly as `hub/coupling.py` owns the
tool modules' faces. A world generator adds an activity with one import and
one call, and knows nothing else about it:

```python
act_body, act_sensor = plate_gate_xml(PLATE_XY, (gx1, GATE_Y), FENCE_HALF_H)
```

Sensors come back separately because MuJoCo wants them in their own top-level
`<sensor>` section.

The supporting pieces in `activity/base.py`:

| piece | for |
|---|---|
| `Activity` | one state machine: `sense()` in, `flags` out |
| `ActivitySet` | all of a world's activities; one step hook, one snapshot |
| `Threshold` | a sensor threshold with hysteresis and optional latching |
| `GeomToggle` | pre-allocated `rgba` / `size` states, selected by name |
| `MocapToggle` | pre-allocated **poses**, for scenery that must move |

---

## 3. The rules

### 3.1 Sense, never assume

The criterion reads contacts or joint sensors. A commanded gate is a belief;
a joint past its threshold is a fact. The plate reads a **`jointpos` sensor**
rather than `qpos` — the same number today, and not the same thing on
hardware, where a real plate has a switch or an encoder. Model the thing the
robot will actually have and the criterion transfers.

### 3.2 Thresholds need hysteresis

A bare threshold on a sprung joint chatters: the plate rings as a wheel rolls
over it, and a single comparison flips on every ring. Measured on the
reference plate, one wheel crossing:

| | flips |
|---|---|
| bare threshold | 4 |
| with hysteresis (6 mm on / 3 mm off) | **2** |

and 2 is the floor, a crossing being one press and one release. Chatter is
not cosmetic: each flip is a telemetry delta, and for a latching effect it is
a decision that cannot be taken back.

### 3.3 Irreversible state latches in Python

"Dug", "planted", "opened" have no restoring force to model. `Threshold(...,
latch=True)` remembers; the physics does not have to. The reference activity
carries one flag of each kind on purpose:

- `pressed` — **live**, true only while something is on the plate;
- `state` — **latched**, `closed` until the first press, then `open` forever.

### 3.4 Visible changes are pre-allocated — and `geom_pos` is a trap

MuJoCo cannot add or remove geometry from a compiled model. So build every
state up front and select between them. **But what you may mutate depends on
where the geom lives**, and this cost an afternoon:

| you want to change | on a jointed body | on static scenery (`body_weldid == 0`) |
|---|---|---|
| `geom_rgba` | ✔ | ✔ |
| `geom_size` | ✔ | ✔ |
| `geom_pos` | ✔ | ✘ **silently does nothing** |
| pose | use the joint | `MocapToggle` |

A body welded to the world — which is *all* scenery — has its geoms' world
poses computed **once**, when `MjData` is created. `mj_kinematics` never
revisits them. Writing `model.geom_pos` lands in the model, changes nothing
anybody reads, and reports no error; the renderer draws `geom_xpos`, which
still holds the compile-time value. Measured: the write took, and `geom_xpos`
was unmoved through `mj_forward`, `mj_step` **and** `mj_setConst`. Writing
`data.geom_xpos` directly does not survive a step either.

The fix is MuJoCo's own mechanism for this: make it a **mocap body**
(`mocap="true"`). No joints, no dynamics, infinite mass, still collides — but
its pose is an *input*, re-read from `data.mocap_pos` on every forward pass.
One attribute in the world, and `MocapToggle` does the rest.

> ⚠ This corrects `rooftop-media-2026/docs/pluggyworld.md`, which lists
> `geom_pos` among the mutable fields without the static-body caveat. Digging
> ("swap a mound geom for a hole visual") happens to be safe because it is
> `rgba`/`size` work — but anything that *moves* needs mocap.

Note the state lives in different places, which matters for the shared world:
geom toggles are **model-global** (every `MjData` sees them), a mocap pose is
**per-`MjData`**. For two robots in one model, the mocap one is correct.

### 3.5 Poll on the step hook, cheaply

`ActivitySet.step_hook(model, data)` gives one callback for a whole world's
activities, appended to `HubMission.step_hooks` — the same per-step seam the
battery drains through and telemetry decimates from. It runs at 500–1000 Hz,
so `sense()` must do no rendering, no allocation and no I/O.

### 3.6 Quantise analogue flags, or leave them off the wire

An analogue value defeats sparse telemetry: it differs almost every step, so
an otherwise-constant activity ships a delta on nearly every frame. Measured
over one wheel crossing (300 frames):

| `depressMm` rounding | telemetry deltas |
|---|---|
| 0.1 mm | 38 |
| 1 mm | **15** |

No viewer can use a tenth of a millimetre. Quantise to what a consumer can
act on, or do not put it on the wire at all.

---

## 4. Telemetry (protocol 0.3.0)

Activity flags ride in the frame, beside `state` and `battery`:

```jsonc
{"t": 123.45, "key": true,
 "robots": {"pluggybot": {...}},
 "world": {"module_lcd": [...]},
 "activities": {"garden_gate": {"state": "open", "pressed": false,
                                "depressMm": 1}}}
```

and the header advertises the names: `"activities": ["garden_gate"]`.

**Sparse, like body poses.** Only activities whose flags changed appear; the
block is omitted entirely when nothing changed. A replayer holds the last
value it saw.

**Re-shipped on every keyframe** — and this matters more here than it does for
poses. An activity's visible effect usually lives on a **static body**: the
gate ships once in the scene description and never again, and its mocap pose
is not in the pose stream at all. So for a change like that **the flag is the
only record anywhere in the stream**. A consumer that missed it has no other
way to learn the gate is open.

⚠ **The emitted-state memory belongs to the sink, not the activity.**
`FrameBuilder` keeps `_last_acts`, exactly where it keeps `_last` for poses.
This was a live bug for ten minutes: `serve.py --record` runs a publisher
*and* a recorder over one physics, each owning a `FrameBuilder` so the sinks
are independent — with the memory on the activity, they consumed each other's
deltas and each shipped a random half of the state changes. Guarded by
`test_two_sinks_over_one_activity_set_stay_independent`.

---

## 5. The build sequence

1. **Decide what the world must remember**, and of what kind: a live flag
   (true while something holds), a latched fact (true forever after), or a
   quantised scalar. That decision is the activity.
2. **Pick the sensed criterion** — which contact or joint sensor answers it —
   and bracket the mechanics so the criterion is reachable but not
   self-triggering. The plate's spring was chosen by arithmetic at both ends:
   its own weight (1.47 N) must not reach the 6 mm trigger, and a drive
   wheel's ~15 N must pass it comfortably. 1200 N/m gives 1.2 mm of sag and
   bottoms out under a wheel. Both ends have margin, and both were checked
   before anything was drawn.
3. **Write the module**: geometry emitter + `Activity` subclass, together.
4. **Add it to a world generator** — one import, one call, and regenerate.
5. **Demo with a filmstrip**, and *sweep the camera* (`ToolPattern.md` §5.4).
   The first cut of the plate demo framed the fence as an undifferentiated
   brown wall in which the gate was invisible in both states — and the gate
   was painted nearly the fence's own colour, so even a good angle would not
   have saved it. A toggle you cannot see in the filmstrip is a toggle you
   cannot debug.
   ⚠ Grab the frame **one step after** the toggle: `sense()` selects it, but
   a mocap pose only reaches `geom_xpos` on the next forward pass. The first
   version photographed a closed gate under a caption saying it was open.
6. **pytest**, per the house rule — every debugged failure becomes an
   assertion shown to fail without its fix. For activities that means:
   the plate rests below its own trigger; pressing latches; releasing does
   *not* unlatch; the robot can actually trip it; the toggle actually moved
   something; and the flags reach a frame, sparsely, and survive a keyframe.
   Pin the `geom_pos` trap as a *defect* test, so the day MuJoCo changes it,
   that is good news wearing a test failure.
7. **Regenerate the protocol fixtures** and update `protocol/README.md`. A
   new flag is content; a new *field* is shape, and shape bumps
   `protocolVersion` — a deliberate two-repo event.

---

## 6. Known gaps

1. **The reference gate blocks nothing the robot needs.** It sits in the
   garden's outer fence, so opening it changes no route. That was chosen to
   keep the world exactly as navigable as it was while the pattern was
   proven. Gating a real passage (the house-to-garden doorway) is the same
   code with the geometry moved — and is the point at which exploration and
   the occupancy grid start to care, which is a question this doc has not
   answered.
2. **Nothing scores an activity yet.** Flags reach the wire; no rule turns
   "the gate opened" into points or into an LLM-visible event. That is the
   evaluation layer in the design doc, and it is the natural next consumer.
3. **The robot has no verb for "operate a mechanism".** The plate is tripped
   by driving over it, which needs no manipulation. A lever or a valve needs
   a claw grip on a fixed mechanism at a known pose — the sink-lever problem
   — and that is a tool-side capability, not an activity-side one.
4. **Activity state is not in the occupancy grid.** A closed gate is a wall
   the planner does not know about; a mocap body that moves is geometry the
   map never re-observes. Fine today because the gate blocks nothing (gap 1).

---

## 7. Checklist

```
[ ] what must the world remember, and is it live / latched / scalar?
[ ] which contact or joint sensor answers it? (sensor, not qpos)
[ ] bracket the mechanics: reachable by the robot, not self-triggering
[ ] hysteresis on every threshold; latch anything irreversible
[ ] effects pre-allocated: rgba/size anywhere, POSE only via MocapToggle
[ ] geometry emitter + Activity subclass in one module under activity/
[ ] one import + one call in the world generator; regenerate the world
[ ] quantise analogue flags to what a consumer can use
[ ] demo with a swept camera, and grab frames one step AFTER a toggle
[ ] pytest: rests-untriggered, latches, robot can trip it, toggle moved,
    flags reach a frame sparsely and survive a keyframe
[ ] every new assertion shown failing without its fix
[ ] SimNotes section; PluggyPlan status; CLAUDE.md entry
[ ] protocol/README.md + fixtures; bump protocolVersion if the SHAPE moved
[ ] MUJOCO_GL=egl uv run pytest -q; uv run ruff check src/ scripts/ tests/
```
