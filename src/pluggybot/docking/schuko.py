"""Standalone Schuko plug/socket contact spike (de-risks milestone 6).

No robot here: a plug rides a compliant carrier and a force-limited servo
pushes it toward a fixed socket. The questions this answers, before any
docking controller depends on them:
  - does the recess funnel actually guide a misaligned plug home in MuJoCo?
  - how much lateral / angular misalignment does insertion forgive?
  - do the contact parameters stay stable, or jitter/explode?

MuJoCo collision geoms must be convex (primitives, or the convex hull of a
mesh), so the concave recess cannot be one geom. It is COMPOSED of convex
pieces instead:
  - well wall: 12 boxes arranged as a dodecagonal ring (the "circle of boxes")
  - entry funnel: 12 more boxes tilted 45 degrees, flaring the mouth outward
  - well floor: 5 slabs leaving two square channels -- the pin holes
  - face plate: 4 boxes framing the mouth, so wide misses hit a wall

Dimensions follow the real Schuko standard where it matters: plug body
35.5 mm dia, recess 37 mm dia x 17 mm deep, pins 4.8 mm dia x 19 mm long,
19 mm apart. Pin tips are capsule hemispheres -- their roundness is the
self-centering chamfer real pins get from machining.

Gravity is off: the robot's lift will own height; this rig isolates the
contact physics from droop.
"""

import math

import mujoco
import numpy as np

# -- geometry (meters), Schuko standard where it matters ----------------------
R_BODY = 0.01775        # plug body radius (35.5 mm dia)
R_WELL = 0.0185         # recess radius (37 mm dia): 0.75 mm clearance per side
WELL_DEPTH = 0.017
# Axial length of the 45-degree entry chamfer. The honest default is 2 mm --
# a real Schuko rim has only a small bevel (the recess WALL is straight, and
# our plug's sharp cylinder edge roughly offsets the real plug's rounded
# nose). Pass a larger value to scene_xml/run_trial to model a dished
# face plate, or to measure what a bigger funnel would buy.
CHAMFER_LEN = 0.002
PIN_SEP = 0.0095        # pin centers at +/-9.5 mm (19 mm apart)
R_PIN = 0.0024          # 4.8 mm dia
PIN_LEN = 0.019
HOLE_HALF = 0.0028      # square pin holes, 5.6 mm: 0.4 mm nominal clearance
N_SEG = 12

# -- the "robot": how hard and how compliantly the plug is pushed -------------
PUSH_FORCE = 10.0       # N cap -- a fraction of the base's 46 N/wheel stall
PUSH_SPEED = 0.02       # m/s commanded approach speed
LAT_STIFFNESS = 150.0   # N/m lateral compliance of the carrier
ROT_STIFFNESS = 1.0     # N*m/rad angular compliance
START_GAP = 0.030       # m between plug face and socket mouth at t=0

SUCCESS_GAP = 0.0025    # m: plug face within this of the well floor = seated


def _ring_boxes(name, x_center, x_half, r_mid, tang_half, rad_half, rgba,
                tilt_deg=0.0):
  """N_SEG boxes forming a ring around the +x axis.

  Each box frame: euler x-rotation phi points its y-axis radially outward;
  a further z-rotation (tilt) leans its long axis outward for the funnel.
  """
  out = []
  for i in range(N_SEG):
    phi = 360.0 * i / N_SEG
    c = math.cos(math.radians(phi))
    s = math.sin(math.radians(phi))
    out.append(
      f'<geom name="{name}{i}" type="box" '
      f'size="{x_half:.5f} {rad_half:.5f} {tang_half:.5f}" '
      f'pos="{x_center:.5f} {r_mid * c:.5f} {r_mid * s:.5f}" '
      f'euler="{phi:.2f} 0 {tilt_deg:.2f}" rgba="{rgba}"/>'
    )
  return "\n      ".join(out)


def socket_geoms_xml(prefix: str = "", chamfer_len: float = CHAMFER_LEN,
                     rgba: str | None = None, solref: str | None = None) -> str:
  """All geoms of one Schuko socket, recess along -x, mouth facing +x.

  prefix namespaces the geom names so several sockets can coexist. rgba
  overrides every geom's color — pass an alpha of 0 for an invisible
  collision layer behind a pretty visual outlet (unrendered geoms also stay
  out of segmentation, so detector ground truth is unaffected). solref, when
  given, is stamped on every geom: room worlds run at timestep 0.002 with no
  spike-style global default, and mm-scale contacts need the stiffer 0.005.
  """
  extra = (f' solref="{solref}"' if solref else "")

  def paint(default):
    # NOTE: returns an UNTERMINATED attribute tail — the caller's template
    # closes the quote. Lets one helper stamp rgba+solref+friction together.
    return (rgba or default) + '"' + extra + ' friction="0.3'

  r_mouth = R_WELL + chamfer_len       # capture radius at the mouth
  seg_tang = (R_WELL + 0.004) * math.tan(math.pi / N_SEG) * 1.2  # overlap
  wall = _ring_boxes(prefix + "wall", -WELL_DEPTH / 2, WELL_DEPTH / 2,
                     R_WELL + 0.002, seg_tang, 0.002, paint("0.8 0.8 0.85 0.4"))
  funnel_half = chamfer_len / math.cos(math.pi / 4) / 2
  funnel = _ring_boxes(prefix + "funnel", chamfer_len / 2, funnel_half,
                       (R_WELL + r_mouth) / 2 + 0.002,
                       r_mouth * math.tan(math.pi / N_SEG) * 1.2, 0.002,
                       paint("0.8 0.8 0.85 0.4"), tilt_deg=45.0)

  # Floor: 5 slabs leaving two 5.6 mm square channels at (y=+/-PIN_SEP, z=0).
  fx, fh = -WELL_DEPTH - 0.010, 0.010          # slab center x, half-thickness
  y_in, y_out = PIN_SEP - HOLE_HALF, PIN_SEP + HOLE_HALF
  mid_z = HOLE_HALF                            # channel row half-height
  floor = f"""
      <geom name="{prefix}floor_top" type="box" size="{fh} 0.03 {(0.03 - mid_z) / 2:.5f}"
            pos="{fx:.5f} 0 {(0.03 + mid_z) / 2:.5f}" rgba="{paint('0.9 0.88 0.85 1')}"/>
      <geom name="{prefix}floor_bot" type="box" size="{fh} 0.03 {(0.03 - mid_z) / 2:.5f}"
            pos="{fx:.5f} 0 {-(0.03 + mid_z) / 2:.5f}" rgba="{paint('0.9 0.88 0.85 1')}"/>
      <geom name="{prefix}floor_mid" type="box" size="{fh} {y_in:.5f} {mid_z:.5f}"
            pos="{fx:.5f} 0 0" rgba="{paint('0.9 0.88 0.85 1')}"/>
      <geom name="{prefix}floor_l" type="box" size="{fh} {(0.03 - y_out) / 2:.5f} {mid_z:.5f}"
            pos="{fx:.5f} {(0.03 + y_out) / 2:.5f} 0" rgba="{paint('0.9 0.88 0.85 1')}"/>
      <geom name="{prefix}floor_r" type="box" size="{fh} {(0.03 - y_out) / 2:.5f} {mid_z:.5f}"
            pos="{fx:.5f} {-(0.03 + y_out) / 2:.5f} 0" rgba="{paint('0.9 0.88 0.85 1')}"/>"""

  # Face plate: 4 boxes framing the mouth so a wide miss hits a wall.
  a = r_mouth + 0.001
  px = chamfer_len + 0.001
  plate = f"""
      <geom name="{prefix}plate_t" type="box" size="0.001 0.055 {(0.055 - a) / 2:.5f}"
            pos="{px:.5f} 0 {(0.055 + a) / 2:.5f}" rgba="{paint('0.9 0.88 0.85 1')}"/>
      <geom name="{prefix}plate_b" type="box" size="0.001 0.055 {(0.055 - a) / 2:.5f}"
            pos="{px:.5f} 0 {-(0.055 + a) / 2:.5f}" rgba="{paint('0.9 0.88 0.85 1')}"/>
      <geom name="{prefix}plate_l" type="box" size="0.001 {(0.055 - a) / 2:.5f} {a:.5f}"
            pos="{px:.5f} {(0.055 + a) / 2:.5f} 0" rgba="{paint('0.9 0.88 0.85 1')}"/>
      <geom name="{prefix}plate_r" type="box" size="0.001 {(0.055 - a) / 2:.5f} {a:.5f}"
            pos="{px:.5f} {-(0.055 + a) / 2:.5f} 0" rgba="{paint('0.9 0.88 0.85 1')}"/>"""
  return wall + "\n      " + funnel + floor + plate


def socket_body_xml(name: str, pos: tuple[float, float, float],
                    facing_deg: float) -> str:
  """One dockable socket as an invisible collision body, for room worlds.

  Place it at the SAME pos/euler as the visual outlet body (outlet_scene's
  convention: facing 0 faces +x, 90 faces +y): the collision mouth then sits
  ~1 mm behind the painted plate face. The recess extends ~37 mm behind the
  mouth — through the 20 mm room wall, poking out the far side, which is
  cosmetic (outside the room, and invisible anyway).
  """
  return f"""
  <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" euler="0 0 {facing_deg:.1f}">
      {socket_geoms_xml(prefix=name + "_", rgba="0.5 0.5 0.5 0", solref="0.005 1")}
  </body>"""


def scene_xml(y_off: float = 0.0, z_off: float = 0.0, yaw_deg: float = 0.0,
              chamfer_len: float = CHAMFER_LEN, noslip: int = 0) -> str:
  """Socket at the origin facing +x; plug carrier approaching from +x.

  y_off/z_off: lateral offset of the approach line (m). yaw_deg: angular
  misalignment of the approach direction (rotation about z). chamfer_len:
  axial length of the 45-degree entry bevel (see CHAMFER_LEN). noslip:
  `noslip_iterations` for the run -- the plug, like the hub peg, seats by
  SLIDING down its entry chamfer, so a global noslip policy (issue #3) has
  to be measured against it, not assumed safe.
  """
  socket = socket_geoms_xml(chamfer_len=chamfer_len)

  # Plug: cylinder body, two capsule pins (fromto excludes the cap radii).
  face = -0.020                                # plug face in plug-body frame
  pin_end = face - PIN_LEN + R_PIN
  pins = "\n      ".join(
    f'<geom name="pin_{lbl}" type="capsule" size="{R_PIN}" '
    f'fromto="{face + R_PIN:.5f} {sy * PIN_SEP:.5f} 0 {pin_end:.5f} {sy * PIN_SEP:.5f} 0" '
    f'rgba="0.75 0.75 0.78 1"/>'
    for lbl, sy in (("l", 1), ("r", -1)))

  plug_x = START_GAP + chamfer_len - face      # world x of plug body origin

  return f"""
<mujoco model="schuko_spike">
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 0"
          noslip_iterations="{noslip}"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <default>
    <geom friction="0.3" solref="0.005 1"/>
  </default>
  <worldbody>
    <light pos="0.2 -0.1 0.3" dir="-0.6 0.3 -0.7"/>
    <light pos="0.1 0.2 0.2" dir="-0.4 -0.7 -0.6"/>
    <camera name="side" pos="0.085 -0.14 0.055" xyaxes="0.855 0.519 0 -0.16 0.264 0.951"/>

    <body name="socket" pos="0 0 0">
      {socket}
    </body>

    <!-- Carrier: an approach rail with force-limited drive; the plug hangs
         off it through soft lateral/angular joints (the compliance a real
         arm + steerable base would provide). The rail's pose IS the
         misalignment under test. -->
    <body name="rail" pos="{plug_x:.5f} {y_off:.5f} {z_off:.5f}" euler="0 0 {yaw_deg:.3f}">
      <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
      <!-- damping = PUSH_FORCE / PUSH_SPEED: with a constant PUSH_FORCE motor
           this yields PUSH_SPEED free approach and exactly PUSH_FORCE at jam.
           (A velocity servo here chatters at the timestep frequency -- the
           SimNotes wheel-armature lesson again: kv it would need is far too
           stiff for the carrier's 90 g. Damping is integrated implicitly by
           implicitfast, so this form is unconditionally stable.) -->
      <joint name="advance" type="slide" axis="-1 0 0"
             damping="{PUSH_FORCE / PUSH_SPEED:.1f}"/>
      <body name="plug" pos="0 0 0">
        <joint name="lat_y" type="slide" axis="0 1 0"
               stiffness="{LAT_STIFFNESS}" damping="4" armature="1e-5"/>
        <joint name="lat_z" type="slide" axis="0 0 1"
               stiffness="{LAT_STIFFNESS}" damping="4" armature="1e-5"/>
        <joint name="rot_z" type="hinge" axis="0 0 1"
               stiffness="{ROT_STIFFNESS}" damping="0.05" armature="1e-5"/>
        <joint name="rot_y" type="hinge" axis="0 1 0"
               stiffness="{ROT_STIFFNESS}" damping="0.05" armature="1e-5"/>
        <geom name="plug_body" type="cylinder" size="{R_BODY} 0.020"
              zaxis="1 0 0" rgba="0.15 0.25 0.6 1"/>
        {pins}
        <site name="plug_face" pos="{face} 0 0" size="0.001"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="push" joint="advance" ctrlrange="0 {PUSH_FORCE}"/>
  </actuator>
</mujoco>"""


def run_trial(y_off: float = 0.0, z_off: float = 0.0, yaw_deg: float = 0.0,
              chamfer_len: float = CHAMFER_LEN,
              sim_time: float = 4.0, n_frames: int = 0, noslip: int = 0,
              ) -> tuple[dict, list[np.ndarray]]:
  """One scripted insertion. Returns (result, frames).

  result: gap_mm (plug face to well floor at the end; ~0 = fully seated),
  success, max_force_n (largest contact force seen -- solver health),
  unstable (plug ever moved faster than 1.5 x the commanded speed x 10).
  """
  model = mujoco.MjModel.from_xml_string(
    scene_xml(y_off, z_off, yaw_deg, chamfer_len, noslip))
  data = mujoco.MjData(model)
  face_site = model.site("plug_face").id
  plug_dofs = [model.joint(n).dofadr[0]
               for n in ("advance", "lat_y", "lat_z")]

  frames = []
  renderer = mujoco.Renderer(model, 360, 480) if n_frames else None
  frame_every = max(1, int(sim_time / model.opt.timestep / max(n_frames, 1)))

  data.ctrl[0] = PUSH_FORCE
  steps = int(sim_time / model.opt.timestep)
  max_force = 0.0
  max_speed = 0.0
  f6 = np.zeros(6)
  for step in range(steps):
    mujoco.mj_step(model, data)
    max_speed = max(max_speed, float(np.max(np.abs(data.qvel[plug_dofs]))))
    for i in range(data.ncon):
      mujoco.mj_contactForce(model, data, i, f6)
      max_force = max(max_force, float(np.linalg.norm(f6[:3])))
    if renderer is not None and step % frame_every == 0 and len(frames) < n_frames:
      renderer.update_scene(data, camera="side")
      frames.append(renderer.render().copy())
  if renderer is not None:
    renderer.close()

  gap = float(data.site_xpos[face_site][0]) - (-WELL_DEPTH)
  return {
    "gap_mm": gap * 1000.0,
    "success": gap < SUCCESS_GAP,
    "max_force_n": max_force,
    "unstable": max_speed > 0.3,
  }, frames


ROOM_SOCKETS = {
  # name: (pos, facing_deg) — MUST match the visual outlet bodies in
  # models/room_1.xml. Positions are SURFACE-MOUNT (German "Aufputz"): the
  # socket origin sits 0.037 m proud of the wall face so the entire recess
  # (17 mm well + 20 mm floor) lives in front of the wall. A flush-mounted
  # socket is impossible here: MuJoCo walls are solid boxes with no holes,
  # and the first docking probe measured the pins bottoming on the wall slab
  # THROUGH the recess interior. Aufputz sockets are real hardware, so this
  # is a modelling choice, not a cheat. Wall faces: west -1.99, south -1.99,
  # east +5.99.
  "socket_a": ((-1.953, 1.0, 0.26), 0.0),
  "socket_b": ((0.5, -1.953, 0.30), 90.0),
  "socket_c": ((5.953, 4.5, 0.38), 180.0),
}


def write_room_sockets(path: str = "models/schuko_sockets.xml") -> None:
  """Regenerate the invisible collision sockets included by room_1.xml."""
  bodies = "".join(socket_body_xml(n, p, f) for n, (p, f) in ROOM_SOCKETS.items())
  with open(path, "w") as fh:
    fh.write(
      "<!-- GENERATED by pluggybot.docking.schuko.write_room_sockets().\n"
      "     Regenerate: uv run python -m pluggybot.docking.schuko\n"
      "     Invisible (alpha 0) collision-only Schuko sockets aligned with the\n"
      "     visual outlet bodies in room_1.xml. Convex box composition; see\n"
      "     schuko.py for why and SimNotes.md for the measured tolerances. -->\n"
      f"<mujocoinclude>{bodies}\n</mujocoinclude>\n")


if __name__ == "__main__":
  write_room_sockets()
  print("wrote models/schuko_sockets.xml")
