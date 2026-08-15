"""The reference activity: a pressure plate that latches a gate open.

The first consumer of the activity pattern (issue #8), and deliberately the
smallest thing that exercises every part of it:

  a SENSED criterion      a sprung slide joint with a `jointpos` sensor, so
                          "is it pressed" is read out of the physics rather
                          than inferred from where the robot was told to go
  HYSTERESIS              the plate rings after a wheel rolls onto it, and a
                          bare threshold turns that ringing into chatter
  a LATCH                 "the gate is open" has no restoring force to model,
                          so Python remembers it and the sim does not have to
  PRE-ALLOCATED TOGGLES   the gate drops into the ground and a lamp goes from
                          red to green -- `geom_pos` and `geom_rgba` on the
                          live model, both states built at load time
  TELEMETRY               two flags on the wire, sparse, re-shipped on
                          keyframes

An activity module owns BOTH its geometry and its state machine, the way
`hub/coupling.py` owns the tool modules' faces. A world generator calls
`plate_gate_xml()` and knows nothing else about it; adding an activity to a
world is one import and one call.

The mechanic is a placeholder for a real one: this gate sits in the garden's
outer fence, so opening it changes no route the robot needs. Gating an
actual passage (the house-to-garden doorway) is the same code with the
geometry moved, and is the obvious next step once something wants to be
kept out.
"""

import math

from pluggybot.activity.base import (
  Activity, GeomToggle, MocapToggle, Threshold,
)

# ---- plate mechanics --------------------------------------------------------
PLATE_HALF = 0.20         # m: a 400 mm pad. Wide enough that arriving a few
                          # cm off still lands a wheel on it -- this is a
                          # centimetre-tolerance target, like the sowing
                          # points, and does not deserve a millimetre
                          # approach controller.
PLATE_THICK = 0.005       # half-thickness of the pad itself
PLATE_REST_Z = 0.016      # m: pad centre at rest, so its top sits 21 mm up.
                          # A step for a 90 mm wheel to climb, and low enough
                          # that it does: measured, not assumed.
PLATE_TRAVEL = 0.010      # m of downward travel. Bottoms out 1 mm clear of
                          # the floor plane at full depression -- a sprung
                          # part that can reach the ground is a contact the
                          # solver has to referee for no reason.
PLATE_MASS = 0.15
PLATE_STIFFNESS = 1200.0  # N/m. Bracketed, not guessed: the pad's own weight
                          # (1.47 N) must NOT reach the trigger depth, and a
                          # drive wheel's load (~15 N) must exceed it easily.
                          # 1.47/1200 = 1.2 mm of static sag against a 6 mm
                          # trigger; 15 N would want 12.5 mm and so bottoms
                          # the plate out at 10 mm. Both ends have margin.
PLATE_DAMPING = 12.0
PLATE_ON = 0.006          # m of depression at which the plate reads pressed
PLATE_OFF = 0.003         # ...and at which it reads released again. The gap
                          # IS the hysteresis, and it is what stops a ringing
                          # sprung joint from chattering the state machine.

# ---- gate -------------------------------------------------------------------
GATE_HALF_LEN = 0.50      # m: fills a 1 m gap in the fence
GATE_THICK = 0.02
GATE_DROP = 1.20          # m the panel sinks when it opens -- clear below the
                          # ground plane, so "open" reads unambiguously in a
                          # camera frame instead of half-overlapping the fence
                          # it slid behind.
LAMP_R = 0.05
LAMP_RED = (0.85, 0.20, 0.18, 1.0)
LAMP_GREEN = (0.25, 0.85, 0.35, 1.0)


def plate_gate_xml(plate_xy: tuple[float, float],
                   gate_xy: tuple[float, float],
                   gate_half_h: float,
                   along_x: bool = False,
                   prefix: str = "garden") -> tuple[str, str]:
  """MJCF for one plate-and-gate activity: (worldbody, sensor).

  `gate_xy` is the centre of the gap the panel fills; `along_x` says which
  way the fence runs there. Returns the two fragments separately because
  MuJoCo wants sensors in their own top-level section.
  """
  px, py = plate_xy
  gx, gy = gate_xy
  hy, hx = ((GATE_HALF_LEN, GATE_THICK) if not along_x
            else (GATE_THICK, GATE_HALF_LEN))
  body = f"""
    <!-- Pressure plate: a sprung pad on a slide joint. The JOINT is the
         sensor; the pad is just what the wheel touches. -->
    <body name="{prefix}_plate" pos="{px:.4f} {py:.4f} {PLATE_REST_Z:.4f}">
      <joint name="{prefix}_plate_joint" type="slide" axis="0 0 1"
             range="{-PLATE_TRAVEL:.4f} 0" stiffness="{PLATE_STIFFNESS}"
             damping="{PLATE_DAMPING}" armature="1e-5"/>
      <geom name="{prefix}_plate_pad" type="box"
            size="{PLATE_HALF:.4f} {PLATE_HALF:.4f} {PLATE_THICK:.4f}"
            mass="{PLATE_MASS}" rgba="0.42 0.44 0.48 1"/>
    </body>
    <!-- Gate panel. A MOCAP body: no joints and no dynamics, so its two
         states are selected rather than simulated -- but mocap poses are
         an INPUT that kinematics re-reads every step, which a welded
         body's geom_pos is not (see activity/base.GeomToggle). It still
         collides, so a closed gate is a real barrier. -->
    <body name="{prefix}_gate" mocap="true"
          pos="{gx:.4f} {gy:.4f} {gate_half_h:.4f}">
      <geom name="{prefix}_gate_panel" type="box"
            size="{hx:.4f} {hy:.4f} {gate_half_h:.4f}"
            rgba="0.62 0.66 0.72 1"/>
    </body>
    <!-- Indicator lamp on the gate post: the same discrete state said in
         colour, which is what makes the toggle legible in a filmstrip. -->
    <body name="{prefix}_lamp" pos="{gx:.4f} {gy + (GATE_HALF_LEN if not along_x else 0):.4f} {gate_half_h * 2 + LAMP_R:.4f}">
      <geom name="{prefix}_lamp_bulb" type="sphere" size="{LAMP_R:.4f}"
            contype="0" conaffinity="0"
            rgba="{' '.join(str(v) for v in LAMP_RED)}"/>
    </body>"""
  sensor = (f'<jointpos name="{prefix}_plate_pos" '
            f'joint="{prefix}_plate_joint"/>')
  return body, sensor


class PlateGate(Activity):
  """Pressure plate -> latched gate.

  Two flags, and they are deliberately different in kind:
    `pressed`  LIVE. True only while something is standing on the plate.
    `state`    LATCHED. "closed" until the plate is first pressed, then
               "open" forever -- the plate is a switch, not a hold-to-open
               button, and a world fact with no restoring force belongs in
               Python rather than in the solver.
  """

  def __init__(self, model, data, prefix: str = "garden",
               name: str | None = None) -> None:
    super().__init__(name or f"{prefix}_gate")
    self.prefix = prefix
    self.sensor_adr = int(model.sensor(f"{prefix}_plate_pos").adr[0])
    self.press = Threshold(on=PLATE_ON, off=PLATE_OFF)
    self.latch = Threshold(on=0.5, latch=True)
    rest = [float(v) for v in model.body_pos[model.body(f"{prefix}_gate").id]]
    self.gate = MocapToggle(model, data, f"{prefix}_gate", {
      "closed": {"pos": rest},
      "open": {"pos": [rest[0], rest[1], rest[2] - GATE_DROP]},
    })
    self.lamp = GeomToggle(model, f"{prefix}_lamp_bulb", {
      "closed": {"rgba": list(LAMP_RED)},
      "open": {"rgba": list(LAMP_GREEN)},
    })
    self.gate.select("closed")
    self.lamp.select("closed")
    self.set(state="closed", pressed=False, depressMm=0.0)

  def depth(self, data) -> float:
    """How far the plate is pushed down, in metres (positive = down).

    Read from the JOINT SENSOR rather than from qpos, which is the same
    number today and will not be on hardware: a real plate has a switch or
    an encoder, and modelling the thing the robot will actually have is
    what keeps the criterion transferable.
    """
    return -float(data.sensordata[self.sensor_adr])

  def sense(self, model, data) -> None:
    d = self.depth(data)
    pressed = self.press.update(d)
    opened = self.latch.update(1.0 if pressed else 0.0)
    state = "open" if opened else "closed"
    self.gate.select(state)
    self.lamp.select(state)
    # Quantised to the whole millimetre on purpose. An ANALOGUE flag defeats
    # sparseness: it differs almost every step, so an otherwise-constant
    # activity would ship a delta on nearly every frame. Measured over one
    # wheel crossing (300 frames): 38 deltas at 0.1 mm, 15 at 1 mm -- and no
    # viewer can use a tenth of a millimetre. Quantise an analogue value to
    # what a consumer can actually act on, or leave it off the wire.
    self.set(state=state, pressed=pressed, depressMm=round(d * 1000))


def plate_center(model, prefix: str = "garden") -> tuple[float, float]:
  """World (x, y) of the plate's pad -- what a demo drives to."""
  bid = model.body(f"{prefix}_plate").id
  return (float(model.body_pos[bid][0]), float(model.body_pos[bid][1]))


def approach_pose(model, prefix: str = "garden",
                  heading: float = 0.0,
                  standoff: float = 0.9) -> tuple[float, float, float]:
  """(x, y, heading) to start a straight run onto the plate from."""
  px, py = plate_center(model, prefix)
  return (px - standoff * math.cos(heading),
          py - standoff * math.sin(heading), heading)
