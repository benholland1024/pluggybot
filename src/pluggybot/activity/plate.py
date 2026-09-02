"""The reference activity: a pressure plate that turns on a light.

The first consumer of the activity pattern (issue #8), and deliberately the
smallest thing that exercises every part of it:

  a SENSED criterion      a sprung slide joint with a `jointpos` sensor, so
                          "is it pressed" is read out of the physics rather
                          than inferred from where the robot was told to go
  HYSTERESIS              the plate rings after a wheel rolls onto it, and a
                          bare threshold turns that ringing into chatter
  a LATCH                 "the light is on" has no restoring force to model,
                          so Python remembers it and the sim does not have to
  PRE-ALLOCATED TOGGLES   the bulb goes from unlit to lit -- `geom_rgba` on
                          the live model, both states built at load time
  TELEMETRY               two flags on the wire, sparse, re-shipped on
                          keyframes

An activity module owns BOTH its geometry and its state machine, the way
`rack/coupling.py` owns the tool modules' faces. A world generator calls
`plate_light_xml()` and knows nothing else about it; adding an activity to a
world is one import and one call.

⚠ THIS USED TO BE A GATE, and the change is a decision, not a cleanup
(issue #93). The gate was designed under an explicit rule -- it blocks no
route the robot needs -- and issue #68 broke that rule by putting the street
behind it: the street became reachable only if a drive happened to clip a
plate 0.34 m off the natural line, and a robot that drove at the shut panel
ground its wheels and pumped 4.38 m of imaginary travel into its odometry in
30 s (issue #94). A plate that turns on a light is also simply a thing
people recognise -- a motion light, pressure-activated -- where a
plate-operated gate is not. The street doorway is now a plain open gap.

The gate was also this repo's one live MOCAP body, and its lesson survives
it: `geom_pos` mutation is silently inert on anything welded to the world,
so a thing that must MOVE between activity states needs a mocap body, whose
pose is an input kinematics re-reads every step. The lesson and its
measurements are in docs/ActivityPattern.md §3.4, and `MocapToggle` stays in
activity/base.py, guarded by its own synthetic-model tests, for the next
activity that needs motion.
"""

import math

from pluggybot.activity.base import Activity, GeomToggle, Threshold

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

# ---- the light --------------------------------------------------------------
#: The bulb sits on a pole BESIDE the plate, garden-path-light style, so the
#: cause and the effect are in one camera frame -- step on the plate, the
#: light next to you comes on. The offset is north (+y): the approach runs
#: east through the garden doorway, so the pole stands clear of the drive
#: line, and 0.8 m keeps its inflation halo (0.35 m) well off the pad the
#: robot must end standing on (issue #92's lesson, applied at design time).
LAMP_OFFSET = (0.0, 0.8)
LAMP_POLE_R = 0.02
LAMP_POLE_HALF_H = 0.45   # a 0.9 m pole: bulb at eye height for a 0.223 m
                          # LIDAR world, i.e. visible in a filmstrip, clear
                          # of the beam
LAMP_R = 0.05
#: Unlit is DIM, not black: a black sphere against the garden reads as a
#: hole in the render, and the robot's cameras see rgba too (the reason
#: hints never ride in colors -- this is honest scenery, not encoding).
LAMP_UNLIT = (0.35, 0.34, 0.30, 1.0)
LAMP_LIT = (1.0, 0.85, 0.35, 1.0)


def plate_light_xml(plate_xy: tuple[float, float],
                    prefix: str = "garden") -> tuple[str, str]:
  """MJCF for one plate-and-light activity: (worldbody, sensor).

  Returns the two fragments separately because MuJoCo wants sensors in
  their own top-level section.
  """
  px, py = plate_xy
  lx, ly = px + LAMP_OFFSET[0], py + LAMP_OFFSET[1]
  bulb_z = LAMP_POLE_HALF_H * 2 + LAMP_R
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
    <!-- The light: a pole beside the plate, bulb on top. The POLE collides
         (a sturdy fixture the robot should map and plan around); the BULB
         does not, like the plants -- it is above the LIDAR's beam anyway,
         and a non-contact sphere is one less thing the solver referees.
         The bulb's two states are rgba toggles on the live model; nothing
         here ever moves, which is why no mocap body is needed (the gate
         this replaced was the mocap example -- ActivityPattern.md 3.4). -->
    <body name="{prefix}_light" pos="{lx:.4f} {ly:.4f} 0">
      <geom name="{prefix}_light_pole" type="cylinder"
            size="{LAMP_POLE_R:.4f} {LAMP_POLE_HALF_H:.4f}"
            pos="0 0 {LAMP_POLE_HALF_H:.4f}" rgba="0.30 0.31 0.33 1"/>
      <geom name="{prefix}_light_bulb" type="sphere" size="{LAMP_R:.4f}"
            pos="0 0 {bulb_z:.4f}" contype="0" conaffinity="0"
            rgba="{' '.join(str(v) for v in LAMP_UNLIT)}"/>
    </body>"""
  sensor = (f'<jointpos name="{prefix}_plate_pos" '
            f'joint="{prefix}_plate_joint"/>')
  return body, sensor


class PlateLight(Activity):
  """Pressure plate -> latched light.

  Two flags, and they are deliberately different in kind:
    `pressed`  LIVE. True only while something is standing on the plate.
    `state`    LATCHED. "off" until the plate is first pressed, then "on"
               forever -- the plate is a switch, not a hold-to-press button,
               and a world fact with no restoring force belongs in Python
               rather than in the solver. (A real motion light times out;
               this one latching is what makes the robot's one visit leave a
               visible mark on the world, which is the activity's point.)
  """

  def __init__(self, model, data, prefix: str = "garden",
               name: str | None = None) -> None:
    super().__init__(name or f"{prefix}_light")
    self.prefix = prefix
    self.sensor_adr = int(model.sensor(f"{prefix}_plate_pos").adr[0])
    self.press = Threshold(on=PLATE_ON, off=PLATE_OFF)
    self.latch = Threshold(on=0.5, latch=True)
    self.lamp = GeomToggle(model, f"{prefix}_light_bulb", {
      "off": {"rgba": list(LAMP_UNLIT)},
      "on": {"rgba": list(LAMP_LIT)},
    })
    self.lamp.select("off")
    self.set(state="off", pressed=False, depressMm=0.0)

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
    lit = self.latch.update(1.0 if pressed else 0.0)
    state = "on" if lit else "off"
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
