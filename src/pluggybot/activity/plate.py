"""The reference activity: a pressure plate that turns on a light.

The first consumer of the activity pattern (issue #8), and deliberately the
smallest thing that exercises every part of it:

  a SENSED criterion      the load on the floor's square under it, a touch
                          sensor (every plate flush with the floor since
                          #684), so "is it pressed" is read out of the
                          physics rather than inferred from where the robot
                          was told to go
  HYSTERESIS              a load rings as a foot lands, and a bare threshold
                          turns that ringing into chatter
  a LATCH                 "the light is on" has no restoring force to model,
                          so Python remembers it and the sim does not have to
  PRE-ALLOCATED TOGGLES   the bulb goes from unlit to lit -- `geom_rgba` on
                          the live model, both states built at load time
  TELEMETRY               two flags on the wire, sparse, re-shipped on
                          keyframes

An activity module owns BOTH its geometry and its state machine. A world
generator calls `plate_light_xml()` and knows nothing else about it; adding
an activity to a world is one import and one call.

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


from pluggybot.activity.base import Activity, GeomToggle, Threshold

# ---- plate mechanics --------------------------------------------------------
PLATE_HALF = 0.20         # m: a 400 mm pad. Wide enough that arriving a few
                          # cm off still lands a foot on it -- this is a
                          # centimetre-tolerance target, and does not deserve
                          # a millimetre approach controller.

# ---- the plate: the floor's own square, on a load cell (#684) -----------------
#: A pad flush with the floor cannot sink under a foot -- the floor is a
#: plane, which holds the foot up over the pad as anywhere -- and at the
#: floor's height the plane takes half of what stands on the pad (measured:
#: a box resting on it read 49-50 % of its weight). So a plate is the
#: floor's own square, read by a touch sensor under whatever stands on it,
#: as load cells under a plate in a recess read it; its pad is paint, drawn
#: and touched by nothing (ActivityPattern.md §3.1). A sprung pad 21 mm
#: proud, the plates' before, stalled the served walker at its edge from a
#: standstill and tripped it walking (SimNotes, "A flight that never got
#: up").
#: ⚠ A sole is read where its contacts are, its corners on a plane: one only
#: part over the plate reads the share its corners inside carry, or none.
LOAD_ZONE_HALF = 0.004    # m: the sensor's zone either side of the floor
LOAD_PAINT_HALF = 0.002   # m: the pad's paint, 4 mm, over a ground's 2 mm
LOAD_ON_N = 100.0         # N on it at which it reads pressed: 10 kg, which
                          # nothing a job sets out weighs, where the served
                          # walker's foot crossing it peaked at 476-853 N
                          # (15 walks, each pressing it)
LOAD_OFF_N = 50.0         # ...and released again: the hysteresis, 71 flips
                          # over those walks where a bare 100 N made 175


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


def plate_xml(plate_xy: tuple[float, float], prefix: str = "garden") -> tuple[str, str]:
  """MJCF for one pressure plate, flush with the floor: (worldbody, sensor).

  The plate on its own, because it is the one mechanism the robot can
  operate by walking (ActivityPattern.md gap 3), and the lab wants three
  of them with nothing lit (activity/cage.py, issue #215). Returns the
  two fragments separately because MuJoCo wants sensors in their own
  top-level section. Names: `<prefix>_plate` (its pad, paint), the zone
  over the floor's square, and its touch sensor `<prefix>_plate_load`,
  which `PlateLight` and `Cage` read (`LOAD_ON_N`).
  """
  px, py = plate_xy
  body = f"""
    <!-- Pressure plate, flush with the floor: the floor's own square, read
         by a touch sensor under whatever stands on it. Its pad is paint. -->
    <body name="{prefix}_plate" pos="{px:.4f} {py:.4f} {LOAD_PAINT_HALF:.4f}">
      <geom name="{prefix}_plate_pad" type="box" contype="0" conaffinity="0"
            size="{PLATE_HALF:.4f} {PLATE_HALF:.4f} {LOAD_PAINT_HALF:.4f}"
            rgba="0.42 0.44 0.48 1"/>
    </body>
    <site name="{prefix}_plate_zone" type="box" group="5" pos="{px:.4f} {py:.4f} 0"
          size="{PLATE_HALF:.4f} {PLATE_HALF:.4f} {LOAD_ZONE_HALF:.4f}"/>"""
  return body, f'<touch name="{prefix}_plate_load" site="{prefix}_plate_zone"/>'


def light_xml(light_xy: tuple[float, float], prefix: str = "garden") -> str:
  """The lamp: a pole with a bulb whose two states are rgba toggles."""
  lx, ly = light_xy
  bulb_z = LAMP_POLE_HALF_H * 2 + LAMP_R
  return f"""
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


def plate_light_xml(plate_xy: tuple[float, float],
                    prefix: str = "garden") -> tuple[str, str]:
  """MJCF for one plate-and-light activity: (worldbody, sensor) -- the
  plate, and the lamp `LAMP_OFFSET` beside it."""
  px, py = plate_xy
  plate, sensor = plate_xml(plate_xy, prefix)
  lamp = light_xml((px + LAMP_OFFSET[0], py + LAMP_OFFSET[1]), prefix)
  return plate + lamp, sensor


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
    self.sensor_adr = int(model.sensor(f"{prefix}_plate_load").adr[0])
    self.press = Threshold(on=LOAD_ON_N, off=LOAD_OFF_N)
    self.latch = Threshold(on=0.5, latch=True)
    self.lamp = GeomToggle(model, f"{prefix}_light_bulb", {
      "off": {"rgba": list(LAMP_UNLIT)},
      "on": {"rgba": list(LAMP_LIT)},
    })
    self.lamp.select("off")
    self.set(state="off", pressed=False)

  def kept_state(self) -> dict:
    return {**super().kept_state(), "pressed": self.press.value,
            "latched": self.latch.value}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    self.flags.pop("depressMm", None)       # a save from the sprung pad's days
    self.press.value = bool(state.get("pressed"))
    self.latch.value = bool(state.get("latched"))
    # the lamp is a geom's colour, which the new process compiled unlit
    self.lamp.current = None
    self.lamp.select("on" if self.latch.value else "off")

  def load(self, data) -> float:
    """The weight on the plate, N: its touch sensor's reading, the normal
    force of what stands on the floor's square under it -- what load cells
    under a plate read, never whose foot it is (`LOAD_ON_N`)."""
    return float(data.sensordata[self.sensor_adr])

  def sense(self, model, data) -> None:
    n = self.load(data)
    pressed = self.press.update(n)
    lit = self.latch.update(1.0 if pressed else 0.0)
    state = "on" if lit else "off"
    self.lamp.select(state)
    # ⚠ No load on the wire (ActivityPattern.md §3.6): a robot standing on
    # the plate shifts tens of newtons a step, so any rounding of it would
    # ship a delta on nearly every frame, and nothing reads one.
    self.set(state=state, pressed=pressed)


def plate_center(model, prefix: str = "garden") -> tuple[float, float]:
  """World (x, y) of the plate's pad -- what a demo drives to."""
  bid = model.body(f"{prefix}_plate").id
  return (float(model.body_pos[bid][0]), float(model.body_pos[bid][1]))


