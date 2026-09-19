"""The lab's cage (issue #215): the experiment zone's props, as geometry.

The mouse's cage is an ACTIVITY (issue #226 builds its state machine: a
mouse whose state -- resting, eating, playing, hiding, on its side -- a
shock changes, and food, a toy and company also change, on their own
clocks). This module is the half an activity module owns FIRST, its
geometry (ActivityPattern.md section 2): the world generator calls
`cage_xml()` and knows nothing else, and #226 adds the `Activity` subclass
here without touching the generator -- which is the whole point of landing
the props in the same world change as the second house (#215: one regime
break, not two).

What the lab holds, and why each thing is the shape it is:

  the CAGE     one body: a solid tray and four upright wall slabs, 0.30 m
               tall so the LIDAR (0.223 m) maps them and the robot plans
               around the enclosure. The website draws the slabs as bars
               (hint `cage`); the marker is what the robot believes.
  the MOUSE    a MOCAP body (`mocap="true"`): the one way scenery MOVES
               between pre-allocated poses (ActivityPattern.md 3.4 --
               `geom_pos` on a welded body silently does nothing). It rests
               at `MOUSE_POSES["resting"]`; #226 selects the others through
               `MocapToggle`. `dynamic: true` on the wire, so a frame
               carries it wherever it goes.
  the BOWL, the WHEEL, the HIDE BOX   what the other states stand at: a
               pose per prop, so "eating" is a position the website can
               draw and a fact the state machine can name.
  three PLATES in a row in front of the cage -- `shock`, `feed`, `toy` --
               each the garden's sprung plate (`plate.plate_xml`), because
               driving onto a plate is the one mechanism this robot can
               operate today (ActivityPattern.md gap 3). Which plate does
               what is #226's; here they are three pads with joint sensors.
               Company is standing near the cage and needs no prop.

The disclosure line, the task and every rule about the mouse are #226's and
belong in the mind's prompt, not here; nothing in this module is shown to
the robot.
"""

from pluggybot.activity.plate import PLATE_HALF, plate_xml

# ---- the cage ---------------------------------------------------------------
CAGE_HALF = (0.30, 0.20)          # the tray, and the enclosure's footprint
CAGE_TRAY_HALF_T = 0.010          # a 20 mm tray the mouse stands on
CAGE_WALL_HALF_T = 0.005          # 10 mm wall slabs (bars, on the website)
CAGE_WALL_HALF_H = 0.15           # 0.30 m walls: above the LIDAR's beam, so
                                  # the enclosure is mapped, not driven into
CAGE_RGBA = "0.72 0.72 0.75 1"

# ---- what stands inside, in the cage's own frame ----------------------------
BOWL_XY, BOWL_R, BOWL_HALF_H = (0.20, -0.10), 0.030, 0.012
WHEEL_XY, WHEEL_R, WHEEL_HALF_T = (0.20, 0.10), 0.060, 0.008
HIDE_XY, HIDE_HALF = (-0.20, 0.10), (0.050, 0.040, 0.030)

# ---- the mouse ----------------------------------------------------------------
MOUSE_R = 0.015                   # a capsule lying along the cage's x
MOUSE_HALF_LEN = 0.020
MOUSE_Z = 2 * CAGE_TRAY_HALF_T + MOUSE_R     # on the tray
MOUSE_RGBA = "0.55 0.48 0.42 1"
#: The five states' poses, in the cage's frame (issue #226 selects them).
#: `on_its_side` rolls the capsule a quarter turn about the cage's x, which
#: is the one pose the geometry can show; the rest are places.
MOUSE_POSES = {
  "resting": {"pos": (-0.20, -0.10, MOUSE_Z)},
  "eating": {"pos": (BOWL_XY[0] - 0.06, BOWL_XY[1], MOUSE_Z)},
  "playing": {"pos": (WHEEL_XY[0] - 0.05, WHEEL_XY[1], MOUSE_Z)},
  "hiding": {"pos": (HIDE_XY[0], HIDE_XY[1], MOUSE_Z)},
  "on_its_side": {"pos": (0.0, -0.10, MOUSE_Z),
                  "quat": (0.7071068, 0.7071068, 0.0, 0.0)},
}
MOUSE_STATES = tuple(MOUSE_POSES)

# ---- the plates ---------------------------------------------------------------
#: In a row SOUTH of the cage (the robot drives north onto one, facing the
#: cage), a metre apart so the inflated planning mask (0.35 m) never merges
#: two pads, and a metre off the cage so a wheel on a pad is not a chassis in
#: the enclosure. Offsets from the cage's centre.
PLATE_OFFSETS = {"shock": (-1.0, -1.2), "feed": (0.0, -1.2), "toy": (1.0, -1.2)}
PLATE_NAMES = tuple(PLATE_OFFSETS)


def cage_xml(cage_xy: tuple[float, float],
             prefix: str = "lab") -> tuple[str, str]:
  """MJCF for the cage, what is in it, and its three plates: (worldbody,
  sensor). Sensors come back separately because MuJoCo wants them in their
  own top-level section."""
  cx, cy = cage_xy
  hx, hy = CAGE_HALF
  wt, wh = CAGE_WALL_HALF_T, CAGE_WALL_HALF_H
  body = f"""
    <!-- The cage: a tray and four wall slabs in ONE body (a hint is per
         body); the website draws the slabs as bars. Solid and mapped. -->
    <body name="{prefix}_cage" pos="{cx:.4f} {cy:.4f} 0">
      <geom name="{prefix}_cage_tray" type="box"
            size="{hx:.4f} {hy:.4f} {CAGE_TRAY_HALF_T:.4f}"
            pos="0 0 {CAGE_TRAY_HALF_T:.4f}" rgba="{CAGE_RGBA}"/>
      <geom name="{prefix}_cage_wall_w" type="box" size="{wt:.4f} {hy:.4f} {wh:.4f}"
            pos="{-(hx - wt):.4f} 0 {wh:.4f}" rgba="{CAGE_RGBA}"/>
      <geom name="{prefix}_cage_wall_e" type="box" size="{wt:.4f} {hy:.4f} {wh:.4f}"
            pos="{hx - wt:.4f} 0 {wh:.4f}" rgba="{CAGE_RGBA}"/>
      <geom name="{prefix}_cage_wall_s" type="box" size="{hx:.4f} {wt:.4f} {wh:.4f}"
            pos="0 {-(hy - wt):.4f} {wh:.4f}" rgba="{CAGE_RGBA}"/>
      <geom name="{prefix}_cage_wall_n" type="box" size="{hx:.4f} {wt:.4f} {wh:.4f}"
            pos="0 {hy - wt:.4f} {wh:.4f}" rgba="{CAGE_RGBA}"/>
    </body>
    <!-- Inside: the bowl, the wheel (standing, axis along y) and the hide
         box. Static and unhinted; the poses the mouse's states stand at. -->
    <body name="{prefix}_bowl" pos="{cx + BOWL_XY[0]:.4f} {cy + BOWL_XY[1]:.4f} 0">
      <geom name="{prefix}_bowl_geom" type="cylinder"
            size="{BOWL_R:.4f} {BOWL_HALF_H:.4f}"
            pos="0 0 {2 * CAGE_TRAY_HALF_T + BOWL_HALF_H:.4f}" rgba="0.85 0.80 0.60 1"/>
    </body>
    <body name="{prefix}_wheel" pos="{cx + WHEEL_XY[0]:.4f} {cy + WHEEL_XY[1]:.4f} 0">
      <geom name="{prefix}_wheel_geom" type="cylinder"
            size="{WHEEL_R:.4f} {WHEEL_HALF_T:.4f}" quat="0.7071068 0.7071068 0 0"
            pos="0 0 {2 * CAGE_TRAY_HALF_T + WHEEL_R:.4f}" rgba="0.60 0.62 0.66 1"/>
    </body>
    <body name="{prefix}_hide" pos="{cx + HIDE_XY[0]:.4f} {cy + HIDE_XY[1]:.4f} 0">
      <geom name="{prefix}_hide_geom" type="box"
            size="{HIDE_HALF[0]:.4f} {HIDE_HALF[1]:.4f} {HIDE_HALF[2]:.4f}"
            pos="0 0 {2 * CAGE_TRAY_HALF_T + HIDE_HALF[2]:.4f}" rgba="0.50 0.35 0.25 1"/>
    </body>
    <!-- The mouse: a MOCAP body, so its pose is an input the activity
         re-writes (ActivityPattern.md 3.4). Rests in its corner. -->
    <body name="{prefix}_mouse" mocap="true"
          pos="{cx + MOUSE_POSES['resting']['pos'][0]:.4f} {cy + MOUSE_POSES['resting']['pos'][1]:.4f} {MOUSE_Z:.4f}">
      <geom name="{prefix}_mouse_body" type="capsule"
            size="{MOUSE_R:.4f} {MOUSE_HALF_LEN:.4f}" quat="0.7071068 0 0.7071068 0"
            rgba="{MOUSE_RGBA}"/>
    </body>"""
  sensors = []
  for name, (dx, dy) in PLATE_OFFSETS.items():
    plate, sensor = plate_xml((cx + dx, cy + dy), prefix=f"{prefix}_{name}")
    body += plate
    sensors.append(sensor)
  return body, "\n    ".join(sensors)


def mouse_poses(cage_xy: tuple[float, float]) -> dict[str, dict]:
  """`MOUSE_POSES` in the WORLD frame, the shape `MocapToggle` takes."""
  cx, cy = cage_xy
  out = {}
  for state, spec in MOUSE_POSES.items():
    px, py, pz = spec["pos"]
    world = {"pos": [cx + px, cy + py, pz]}
    if "quat" in spec:
      world["quat"] = list(spec["quat"])
    out[state] = world
  return out


def cage_center(model, prefix: str = "lab") -> tuple[float, float]:
  """World (x, y) of the cage -- what a demo or a test drives to."""
  bid = model.body(f"{prefix}_cage").id
  return (float(model.body_pos[bid][0]), float(model.body_pos[bid][1]))


def plate_center(model, name: str, prefix: str = "lab") -> tuple[float, float]:
  """World (x, y) of one of the lab's plates (`shock`, `feed`, `toy`)."""
  bid = model.body(f"{prefix}_{name}_plate").id
  return (float(model.body_pos[bid][0]), float(model.body_pos[bid][1]))


#: A wheel on a pad, for the tests: the pad's half-width less a wheel's.
PLATE_REACH = PLATE_HALF
