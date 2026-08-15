"""The activity layer: task state machines over the physics (issue #8).

PluggyWorld has three layers, and MuJoCo only owns one of them:

  1. RIGID BODIES AND CONTACTS -- the only thing MuJoCo simulates. Tools,
     levers, chunky props, a scoop tip touching a soil bed.
  2. TASK STATE MACHINES -- *this* module. Python watching the physics and
     flipping discrete world state: this plot is dug, this seed is planted,
     the gate is open. Contacts and joint sensors in; booleans and scalars
     out.
  3. BROWSER VISUALS -- everything continuous or organic: water pouring,
     dirt, plants growing. Keyed to streamed state flags and never present
     in MuJoCo at all.

The question is never "can MuJoCo model soil, water or growth". It is "can
the robot perform a convincing rigid-body gesture at a known place that a
state machine can VERIFY" -- and that verification is what lives here.

This is not a new idea in the repo, it is the oldest one generalised.
Charging has always worked exactly this way: `rack_charge_contact` is a
contact-derived criterion and the battery filling is bookkeeping. So is
`module_power_contact`, and `ClawTool.holding`, and `pen_on_board`. Every
one of them answers a question about the world with a FACT read out of the
physics rather than with a belief about what was commanded. An activity is
that pattern given a name, a memory, and a way onto the wire.

Four rules the base class exists to enforce:

  SENSE, NEVER ASSUME. The criterion reads contacts or joint sensors. A
  commanded gate is a belief; a joint past its threshold is a fact. This
  repo has paid for that lesson at least five times.

  THRESHOLDS NEED HYSTERESIS. A bare threshold on a sprung joint chatters:
  the plate rings as a wheel rolls over it, and a single comparison flips
  state on every ring. Measured on the reference plate, one wheel crossing:
  **4 flips bare, 2 with hysteresis** -- and 2 is the floor, since a
  crossing is one press and one release. Chatter is not cosmetic: each flip
  is a telemetry delta, and for a latching effect it is a decision that
  cannot be taken back.

  IRREVERSIBLE STATE LATCHES IN PYTHON, NOT IN PHYSICS. "Dug" and "planted"
  have no restoring force to model. The state machine remembers; the sim
  does not have to.

  VISIBLE STATE CHANGES ARE PRE-ALLOCATED. MuJoCo cannot add or remove
  geometry at runtime, but `geom_pos` / `geom_size` / `geom_rgba` on the
  live model are all mutable. So build every state up front and select
  between them (`GeomToggle`). This costs nothing per step, unlike
  deformable bodies, and it keeps the ROBOT's own cameras honest: the sim
  sees the same world the website is told about.
"""

from typing import Callable


class GeomToggle:
  """A geom with pre-built states, selected by index.

  Every state is computed when the toggle is constructed, so selecting one
  is a couple of array writes and no allocation -- and, crucially, no
  recompilation: MuJoCo cannot add or remove geometry from a compiled model
  at all.

  ⚠ **`pos` DOES NOT WORK ON STATIC SCENERY, silently.** A body welded to
  the world (`body_weldid == 0` -- i.e. anything with no joints hanging off
  worldbody, which is all scenery) has its geoms' world poses computed ONCE
  when `MjData` is created. `mj_kinematics` never revisits them, so writing
  `model.geom_pos` lands in the model, changes nothing anybody reads, and
  reports no error. Measured on the reference gate: the write took, and
  `geom_xpos` sat at its compile-time value through `mj_forward`,
  `mj_step`, and `mj_setConst` alike -- and the renderer draws `geom_xpos`.
  Writing `data.geom_xpos` directly does not survive either.

  So:
    * `rgba` and `size` -- fine anywhere. Both are read live by the
      renderer and by collision, with no world-pose cache in between.
    * `pos` -- only on a body that is genuinely part of the kinematic tree.
      To MOVE a piece of scenery, use `MocapToggle`, which is MuJoCo's
      supported mechanism for exactly this.

  This corrects the PluggyWorld design doc, which lists `geom_pos` among
  the mutable fields without the static-body caveat.
  """

  def __init__(self, model, geom: str, states: dict[str, dict]) -> None:
    """`states` maps a state name to any of pos / size / rgba."""
    self.model = model
    self.gid = model.geom(geom).id
    self.geom = geom
    self.states = states
    self.current: str | None = None

  def select(self, state: str) -> None:
    if state == self.current:
      return
    if state not in self.states:
      raise KeyError(f"{self.geom}: no pre-allocated state {state!r} "
                     f"(have {sorted(self.states)})")
    spec = self.states[state]
    if "pos" in spec:
      self.model.geom_pos[self.gid] = spec["pos"]
    if "size" in spec:
      self.model.geom_size[self.gid] = spec["size"]
    if "rgba" in spec:
      self.model.geom_rgba[self.gid] = spec["rgba"]
    self.current = state


class MocapToggle:
  """A MOCAP body with pre-built poses, selected by name.

  The way to move scenery. A mocap body is static -- no joints, no
  dynamics, infinite mass, and it still collides -- but its pose is an
  INPUT, read from `data.mocap_pos` / `mocap_quat` on every forward pass.
  That is precisely the "pre-allocated states you select between" idea,
  except MuJoCo actually recomputes the world pose for it, which it does
  not for a welded body's `geom_pos` (see GeomToggle).

  Costs one line in the world: `<body name="..." mocap="true">`. Note the
  pose lives in `data`, not `model`, so two MjData over one model can hold
  the gate open and closed independently -- which is the correct behaviour
  and the opposite of the geom toggles, whose state is model-global.
  """

  def __init__(self, model, data, body: str, states: dict[str, dict]) -> None:
    self.model, self.data = model, data
    self.bid = model.body(body).id
    self.mocapid = int(model.body_mocapid[self.bid])
    if self.mocapid < 0:
      raise ValueError(
        f"{body} is not a mocap body -- add mocap=\"true\" to it. Without "
        "that, moving static scenery silently does nothing (GeomToggle).")
    self.body = body
    self.states = states
    self.current: str | None = None

  def select(self, state: str) -> None:
    if state == self.current:
      return
    if state not in self.states:
      raise KeyError(f"{self.body}: no pre-allocated state {state!r} "
                     f"(have {sorted(self.states)})")
    spec = self.states[state]
    if "pos" in spec:
      self.data.mocap_pos[self.mocapid] = spec["pos"]
    if "quat" in spec:
      self.data.mocap_quat[self.mocapid] = spec["quat"]
    self.current = state


class Activity:
  """One task state machine: sense the physics, own some discrete state.

  Subclasses implement `sense()`, which is called once per physics step and
  must be cheap -- it runs at 500-1000 Hz on the same seam the battery
  drains through, so no rendering, no allocation, no I/O.

  `flags` is the activity's whole public surface: a small dict of JSON-ready
  scalars that goes to telemetry verbatim. Keep it to what a viewer or a
  scoring rule actually needs; it is re-shipped on every keyframe.
  """

  #: telemetry key; must be unique across the world
  name: str = "activity"

  def __init__(self, name: str | None = None) -> None:
    if name is not None:
      self.name = name
    self.flags: dict = {}
    self.changes = 0
    self.on_change: list[Callable[[str, dict], None]] = []

  # ---- subclass surface ----------------------------------------------------

  def sense(self, model, data) -> None:
    """Read the physics and update `self.flags`. Called every step."""
    raise NotImplementedError

  # ---- plumbing ------------------------------------------------------------

  def set(self, **flags) -> None:
    """Update flags, firing `on_change` when any of them actually moved."""
    changed = {k: v for k, v in flags.items() if self.flags.get(k) != v}
    if not changed:
      return
    self.flags.update(changed)
    self.changes += 1
    for hook in self.on_change:
      hook(self.name, dict(self.flags))

  # NOTE: an activity deliberately does NOT remember what has been sent.
  # Sparse-emission bookkeeping lives in the FrameBuilder, exactly where the
  # equivalent bookkeeping for body poses lives, and for a reason that was a
  # live bug for about ten minutes: `serve.py --record` runs a publisher AND
  # a recorder over the same physics, each owning its own FrameBuilder so
  # the two sinks are independent. Had the "already emitted" memory sat on
  # the activity, the two would have consumed each other's deltas and each
  # sink would have shipped a random half of the state changes.


class Threshold:
  """A sensor threshold with hysteresis and a latch, the two things a bare
  comparison is always missing.

  `on` is the level at which the criterion becomes true; `off` is where it
  becomes false again, and must be slacker. Between them the answer is
  whatever it already was, which is what stops a ringing sprung joint from
  chattering (measured: 8 flips per crossing without, 1 with).

  `latch=True` makes the true state permanent -- for "dug", "planted",
  "opened": world facts with no restoring force to model, remembered in
  Python because the physics has no reason to.
  """

  def __init__(self, on: float, off: float | None = None,
               latch: bool = False) -> None:
    self.on = on
    self.off = on if off is None else off
    self.latch = latch
    self.value = False
    # No validation here on purpose: every ordering is meaningful. on == off
    # is a plain threshold with no hysteresis, on > off triggers on a rising
    # signal, on < off on a falling one, and `update` reads the direction
    # off that comparison.

  def update(self, x: float) -> bool:
    if self.latch and self.value:
      return True
    if self.on >= self.off:                     # triggers on rising x
      self.value = x >= self.on if not self.value else x > self.off
    else:                                       # triggers on falling x
      self.value = x <= self.on if not self.value else x < self.off
    return self.value


class ActivitySet:
  """Every activity in one world, polled together and exported together.

  Owns the step hook so a world's activities cost exactly one callback on
  the mission's `step_hooks`, whatever their number.
  """

  def __init__(self, activities=()) -> None:
    self.activities = list(activities)

  def add(self, activity: Activity) -> Activity:
    self.activities.append(activity)
    return activity

  def __iter__(self):
    return iter(self.activities)

  def __len__(self) -> int:
    return len(self.activities)

  def step_hook(self, model, data) -> Callable[[], None]:
    """A zero-argument callback for `HubMission.step_hooks`."""
    def hook() -> None:
      for act in self.activities:
        act.sense(model, data)
    return hook

  def snapshot(self) -> dict:
    """{name: flags} for every activity: the whole world state at once.

    The only export there is. Each telemetry sink diffs this against what it
    last sent, so two sinks over one world stay independent -- see the note
    on Activity about why the memory does not live here.
    """
    return {a.name: dict(a.flags) for a in self.activities}

  @property
  def names(self) -> list[str]:
    return [a.name for a in self.activities]
