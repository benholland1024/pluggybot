"""The quadruped's arm, its coupling and the rack for legs (issue #378).
SimNotes, "The quadruped's arm", has the tables; `scripts/arm_spike.py`
flies them. These pin the rules the tables settled, each as cheaply as it
can still fail for the right reason: arithmetic where the rule is
arithmetic, one short flight where it is the physics."""

from dataclasses import replace
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from pluggybot.legs import arm as am
from pluggybot.legs import model as qm
from pluggybot.legs import rack as rk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import arm_spike as sp  # noqa: E402


def _body(spec=am.ArmSpec()):
  model = mujoco.MjModel.from_xml_string(qm.body_xml(qm.CHOSEN, arm=am.arm_mjcf(spec)))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def _at_bay(rig_kw=None, **lay):
  """A robot standing at the bay after a short settle, the rack laid off by
  `lay` (dy, dyaw); the spike's placed capture, one row."""
  rig = sp.Rig(**(rig_kw or {}))
  rig.hold(1.0)
  rig.lay_rack(**lay)
  return rig


# ---- the arm ------------------------------------------------------------------------

def test_the_parallelogram_keeps_the_plate_at_the_torsos_angle_while_the_arm_moves():
  # The level linkage is an equality on shoulder + elbow + wrist: swung
  # from its stow to full reach and to the floor under gravity, the end
  # plate never leaves the torso's angle. Without it ("body"), the plate
  # takes the joints' sum.
  spec = am.ArmSpec()
  model, data = _body(spec)
  arm = am.ArmDriver(model, data, spec)
  plate = model.body("arm_plate").id
  worst = 0.0
  for q in ((0.0, 0.0), (1.2, -2.2), (-0.6, -1.4)):
    arm.aim(*q)
    for _ in range(int(1.2 / model.opt.timestep)):
      arm.step()
      mujoco.mj_step(model, data)
      rel = data.xmat[1].reshape(3, 3).T @ data.xmat[plate].reshape(3, 3)
      worst = max(worst, abs(math.atan2(rel[0, 2], rel[0, 0])))
  assert worst < math.radians(0.5)
  assert am.plate_angle(spec.with_(level="body"), 1.2, -2.2) == pytest.approx(-1.0)


def test_the_ik_keeps_a_shoulder_past_pi_inside_its_range():
  # Close over the shoulder and behind it, the elbow-up solution wants the
  # shoulder past 180 deg (the stow is at 180); wrapped to -180 it fell out
  # of the joint's range and the arm swung the other way.
  spec = am.ArmSpec()
  x, z = am.wrist_xz(spec, 3.3, -2.9)
  q = am.solve(spec, x, z, "up")
  assert q is not None and q[0] > math.pi
  assert am.wrist_xz(spec, *q) == pytest.approx((x, z), abs=1e-9)
  # ...and with `near`, the branch nearest where the arm is (both exist
  # only near full reach: the elbow bends back to 0.4 rad at most).
  x, z = spec.shoulder_x + 0.59, spec.shoulder_z
  up, down = am.solve(spec, x, z, "up"), am.solve(spec, x, z, "down")
  assert None not in (up, down) and up != down
  assert am.solve(spec, x, z, near=up) == up
  assert am.solve(spec, x, z, near=down) == down


def test_the_level_linkage_carries_a_tools_lean_to_the_torso_not_the_elbow():
  # Both motors sit at the shoulder: the elbow's drives the forearm's
  # ABSOLUTE angle and holds the weight at the wrist about the elbow; what
  # a tool does ahead of the wrist goes to the torso through the level
  # parallelogram. The driver's feed-forward (the sim's own Jacobians)
  # and the closed form agree on what a tool adds, and the elbow's share
  # is NOT the tool's whole moment about the elbow.
  spec = am.ArmSpec()
  model, data = _body(spec)
  arm = am.ArmDriver(model, data, spec)
  qs, qe = 0.4, -0.9
  a = model.jnt_qposadr
  data.qpos[a[model.joint("arm_shoulder").id]] = qs
  data.qpos[a[model.joint("arm_elbow").id]] = qe
  data.qpos[a[model.joint("arm_wrist").id]] = -(qs + qe)
  mujoco.mj_forward(model, data)
  kg, com = 0.6, (0.055, -0.08)
  bare = arm.gravity()
  arm.payload = (kg, com)
  added = arm.gravity() - bare
  closed = (np.array(am.gravity_torques(spec, qs, qe, tool_kg=kg, tool_x=com[0], tool_z=com[1]))
            - np.array(am.gravity_torques(spec, qs, qe)))
  assert added == pytest.approx(closed, abs=1e-6)
  ex, _ = am.elbow_xz(spec, qs)
  px, _ = am.seat_xz(spec, qs, qe)
  assert added[1] < kg * 9.81 * (px + com[0] - ex) - 0.5


def test_the_arm_holds_the_envelopes_heaviest_tool_and_a_payload_inside_its_rating():
  # The envelope's heaviest tool with the bench's heaviest cube, its CoM at
  # the envelope's lever, the arm straight out: the motors hold it under
  # 60 % of the GIM8108-8's continuous rating.
  spec = am.ArmSpec()
  x = spec.shoulder_x + spec.reach + spec.fork.vertex_x - 0.002
  q = am.solve_seat(spec, x, spec.shoulder_z + spec.fork.vertex_z + 0.0042)
  ts, te = am.gravity_torques(spec, *q, tool_kg=am.TOOL_MAX_KG + sp.CUBE_KG,
                              tool_x=am.TOOL_MAX_AHEAD_M, tool_z=-0.08)
  assert max(abs(ts), abs(te)) < 0.6 * spec.motor.rated_torque
  assert am.TOOL_MAX_KG * 9.81 * am.TOOL_MAX_AHEAD_M < am.TOOL_MAX_MOMENT_NM


# ---- the sensors ----------------------------------------------------------------------

@pytest.mark.parametrize("stow, hides", [(am.ArmSpec().stow, False), ((math.pi, -math.pi), True)])
def test_the_stowed_arm_hides_nothing_from_the_lidar_or_the_nose_camera(stow, hides):
  # Folded with the forearm 10 deg up, the fork sits above the nose
  # camera's view and under the LIDAR's plane. Folded flat, it hid 13 % of
  # the nose camera's image.
  rig = sp.Rig(spec=am.ArmSpec(stow=stow), rack=None)
  rig.arm.aim(*stow)
  a = rig.model.jnt_qposadr
  rig.data.qpos[a[rig.model.joint("arm_shoulder").id]] = stow[0]
  rig.data.qpos[a[rig.model.joint("arm_elbow").id]] = stow[1]
  rig.data.qpos[a[rig.model.joint("arm_wrist").id]] = -(stow[0] + stow[1])
  rig.data.qpos[rig.tool_adr:rig.tool_adr + 3] = [5.0, 5.0, 0.1]
  mujoco.mj_forward(rig.model, rig.data)
  o = sp.occlusion(rig, cams=("nav_eye",))
  if hides:
    assert o["nav_eye"] > 0.10
  else:
    assert o["lidar"] == 0 and o["nav_eye"] == 0.0


def test_a_carried_tool_hangs_clear_of_the_lidars_scan_plane():
  # At the carry pose the envelope's longest pendant (the claw's 0.2 m)
  # keeps over the scan plane (the LIDAR site's height over the torso's
  # centre, read off the model).
  model, data = _body()
  lidar_z = data.site_xpos[model.site("lidar").id][2] - data.xpos[1][2]
  seat_z = am.CARRY[1] + am.ForkSpec().seat_rise()
  assert seat_z - am.TOOL_MAX_DROP_M > lidar_z + 0.01


def test_a_carried_tool_is_terrain_to_the_height_scan_unless_it_is_filtered():
  # The policies' height scan reads group-0 geometry as terrain: a tool
  # held over the nose there reads as a 0.4 m obstacle. The robot filters
  # what it carries as it filters its own body.
  from pluggybot.legs.policy import PolicyDriver, WalkingPolicy
  heights = {}
  for group in (0, 3):
    rig = sp.Rig(rack=None, tool_group=group)
    rig.arm.aim(*am.solve_vertex(rig.spec, *am.CARRY))
    rig.hold(0.8)
    rig.mount_tool()
    drv = PolicyDriver(rig.model, rig.data, WalkingPolicy())
    # The scan is the body's height over the terrain: under a tool held
    # over the nose, NEGATIVE by the tool's height over the body.
    heights[group] = -float(np.min(drv.height_scan()))
  assert heights[0] > 0.3 and heights[3] < 0.0


# ---- the coupling --------------------------------------------------------------------

def test_a_tool_facing_the_robot_is_powered_through_its_own_poles():
  # A racked tool faces the robot, so its left conductor rides the arm's
  # right V; the V's are named for the pole they take. Turned round, each
  # conductor meets the other pole's V: a short, not a tool.
  rig = sp.Rig(rack=None)
  rig.arm.aim(*am.solve_vertex(rig.spec, *am.CARRY))
  rig.hold(0.8)
  rig.mount_tool()
  rig.hold(0.3)
  assert rig.powered()
  q = rig.data.qpos[rig.tool_adr + 3:rig.tool_adr + 7].copy()
  rig.data.qpos[rig.tool_adr + 3:rig.tool_adr + 7] = [-q[3], q[2], -q[1], q[0]]   # +pi about z
  rig.data.qvel[rig.tool_dof:rig.tool_dof + 6] = 0.0
  rig.hold(0.3)
  assert not rig.powered()


@pytest.mark.parametrize("intrude, plumb", [(am.ForkSpec().pad_intrude, True), (0.0027, False)])
def test_the_lean_pad_stands_behind_a_seated_tool(intrude, plumb):
  # At the rover's 2.7 mm the pad met the tool plate's bottom edge while
  # the peg was still on the trays and levered it 15 deg up the V's flank.
  spec = am.ArmSpec(fork=replace(am.ForkSpec(), pad_intrude=intrude))
  rig = _at_bay({"spec": spec})
  px, pz = sp.aim_of(rig)
  rig.fork_to(px - am.STANDOFF, pz - am.FORK_DROP, speed=0.2)
  rig.fork_to(px, pz - am.FORK_DROP)
  rig.arm.payload = (rig.tool_kg, sp.TOOL_COM)
  rig.fork_to(px, pz - am.FORK_DROP + am.LIFT)
  assert rig.powered()
  assert (rig.tool_tilt() < 2.0) is plumb


def test_the_end_ramps_push_a_peg_along_harder_than_the_far_v_holds_it():
  # A peg end on a ramp slides in along its axis only if the ramp's push
  # beats the far V's grip. Half the tool's weight on each: a ramp at a
  # pushes N (sin a - mu cos a) with N cos a = W/2, and the far V's flanks
  # at f hold it with mu_peg W / (2 cos f). With 45 deg V's and ramps at the
  # peg's own 0.4 that was 0.30 against 0.28 of W, and a pick from 15 mm
  # off at 4 deg left the end on the ramp, its pole open. The 60 deg V
  # grips harder: the 45 deg ramp's slippery face and the 53 deg ramp's
  # rough one each fall short of the margin, and the ramp is both.
  f = am.ForkSpec()
  from pluggybot.rack.coupling import PEG_FRICTION

  def push(mu, a):
    return 0.5 * (math.tan(a) - mu)
  grip = PEG_FRICTION * 0.5 / math.cos(math.radians(f.flank_deg))
  a = math.atan2(f.ramp_h, f.ramp_w)
  assert push(am.RAMP_MU, a) > 1.4 * grip
  assert push(PEG_FRICTION, a) < 1.2 * grip
  assert push(am.RAMP_MU, math.radians(45.0)) < 1.1 * grip


def test_the_rovers_fork_on_the_arm_takes_nothing_10_mm_off():
  # The PREMISE of the redesign: the rover's end-stops sit 4 mm past its
  # peg's ends, and 10 mm off one lands under the peg.
  rig = _at_bay({"rover": True}, dy=0.010)
  assert not rig.pick(sp.aim_of(rig))["picked"]


@pytest.mark.parametrize("narrow, hung", [(False, True), (True, False)])
def test_a_tool_taken_12_mm_off_hangs_back_between_its_trays(narrow, hung):
  # Hung back from where it was taken, 12 mm off (the walk-in stops up to
  # 15 mm off), a tool's plate must fit between the trays. At +-35 mm (9 mm
  # of room) it set its plate on a tray's corner; at +-45 (19 mm) it hangs
  # back plumb. The stance settles as the capture table's does: its yaw
  # creep carries the fork a few mm while it works.
  kw = {"spec": am.ArmSpec().with_(fork=sp.NARROW_FORK), "rack": sp.NARROW_RACK} if narrow else {}
  rig = sp.settled(**kw)
  rig.lay_rack(dy=-0.012)
  aim = sp.aim_of(rig)
  assert rig.pick(aim)["picked"]
  assert rig.put_back(aim)["returned"] is hung


def test_the_lift_carries_the_peg_clear_of_the_trays_corners():
  # The fork meets a hanging peg FORK_DROP less its seat's rise into the
  # lift, so a carried peg rides LIFT - FORK_DROP + rise over its rest, and
  # its underside must clear the trays' V corners (their flanks' tops,
  # 15.6 mm over a vertex) by more than an aim 10 mm low: the rover's
  # 36 mm cleared them by 4, and such a return knocked the tool off.
  corner = 2 * rk.V_HALF_LEN / math.sqrt(2) - rk.TRAY_VERTEX_DROP

  def clear(lift, drop, fork):
    return lift - drop + fork.seat_rise() - (corner + rk.PEG_R)
  assert clear(am.LIFT, am.FORK_DROP, am.ForkSpec()) >= 0.013
  assert clear(0.036, 0.022, sp.ROVER_FORK) < 0.010


def test_the_steeper_v_keeps_the_rovers_mouth():
  # A V's mouth (a flank's run across the plate) is its capture along the
  # bay. Steepened to 60 deg on the rover's 22 mm flanks it narrowed from
  # 15.6 mm to 11, and the capture's corners went chaotic: a pick 15 mm
  # across at 4 deg failed after a 0.4 mm move of the end-ramps. Lengthened
  # to keep 15.6, every corner to +-18 mm and +-4 deg passes.
  def mouth(f):
    return 2 * f.v_half_len * math.cos(math.radians(f.flank_deg))
  assert mouth(am.ForkSpec()) >= mouth(sp.ROVER_FORK) - 1e-4
  assert mouth(replace(am.ForkSpec(), v_half_len=sp.ROVER_FORK.v_half_len)) < 0.012


def test_the_fork_passes_under_a_hanging_peg_on_the_way_in():
  # The V's flanks' tops pass under a hanging peg's underside with the
  # 3.4 mm the rover's 45 deg V had at a 22 mm drop; the 60 deg flanks
  # stand 27 mm tall, and at that drop they would meet the peg.
  f = am.ForkSpec()
  assert am.FORK_DROP - f.flank_top() - rk.PEG_R >= 0.003
  assert 0.022 - f.flank_top() - rk.PEG_R < 0
  assert 0.022 - sp.ROVER_FORK.flank_top() - rk.PEG_R >= 0.003


def _on_a_pitched_fork(fork: am.ForkSpec, up_flank: float, seconds: float = 0.5) -> float:
  """The fork alone, pitched nose-down to a descent's worst
  (`STAIR_PITCH_DEG`), a tool's peg set `up_flank` up the flank that lays
  down, the tool plumb: how far up that flank the peg is `seconds` later."""
  from pluggybot.rack.coupling import PEG_ABOVE_BODY
  spec = am.ArmSpec().with_(fork=fork)
  a = math.radians(am.STAIR_PITCH_DEG)
  rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
  base = np.array([0.0, 0.0, 0.5])
  f = math.radians(fork.flank_deg)
  seat = np.array([fork.vertex_x, 0.0, fork.vertex_z + fork.seat_rise()])
  up = np.array([math.cos(f), 0.0, math.sin(f)])
  peg = base + rot @ (seat + up_flank * up)
  xml = f"""<mujoco><option timestep="0.002"/>
    <default>{am.ARM_DEFAULTS.format(friction=0.01, tube=am.TUBE_R)}
      {rk.tool_default("tool")}</default>
    <worldbody>
      <body name="arm_plate" pos="{base[0]} 0 {base[2]}"
            quat="{math.cos(a / 2)} 0 {math.sin(a / 2)} 0">
        {am.fork_xml(spec)}
      </body>
      {rk.tool_xml("tool", tuple(peg), yaw=math.pi)}
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
  tool = model.body("tool").id
  axis = data.xpos[tool] + data.xmat[tool].reshape(3, 3) @ [0, 0, PEG_ABOVE_BODY]
  return float((rot.T @ (axis - base) - seat) @ up)


@pytest.mark.parametrize("fork, home", [(am.ForkSpec(), True), (sp.FIRST_FORK, False)])
def test_a_peg_jolted_up_the_low_flank_on_the_stairs_slides_home(fork, home):
  # Down the stairs the plate pitches with the torso and one flank of each
  # V lays down: at the rover's 45 deg, to 12 deg off level at the worst
  # pitch, under the peg's friction angle (21.8), so a jolt that hops the
  # peg 5 mm up it leaves it there, one step from off (flown: 13 of 17
  # tools 60 mm ahead of their peg lost); at 60 deg it slides home.
  assert (_on_a_pitched_fork(fork, 0.005) < 0.001) is home


def test_the_fork_and_the_trays_share_one_peg_with_room_between():
  # The fork's V's take the peg outboard of the trays, under its
  # conductors, with the ramps' capture of room between a fork V and a
  # tray; the stops hold the tool to 1.5 mm along the peg.
  f, r = am.ForkSpec(), rk.DEFAULT
  gap = (f.fork_y - f.v_half_w) - (r.tray_y + rk.TRAY_HALF_W)
  assert gap >= f.ramp_w + (f.stop_y - r.peg_half)
  assert rk.PEG_INSUL_HALF < f.fork_y - f.v_half_w and f.fork_y + f.v_half_w < r.peg_half
  assert f.stop_y - r.peg_half == pytest.approx(0.0015)
  # ...and a racked tool's plate has more room between its trays than the
  # walk-in leaves across (LINEUP_ACROSS).
  from pluggybot.rack.coupling import TOOL_HALF_Y
  assert r.tray_y - rk.TRAY_HALF_W - TOOL_HALF_Y > rk.LINEUP_ACROSS


# ---- falls ---------------------------------------------------------------------------

@pytest.mark.parametrize("reflex", [True, False])
def test_a_robot_pushed_over_with_a_tool_throws_it_and_stands_arm_folded_or_not(reflex):
  # Pushed over while trotting with a tool, it throws the tool either way
  # (a gravity seat cannot hold it upside down). Its arm left out at the
  # carry pose props it on its side: #377's get-up could not roll it, which
  # is why the arm folds as the torso passes 60 deg (the fall reflex).
  # #389's rolls it either way; a get-up that cannot makes the fold
  # necessary again.
  r = sp.fall(reflex=reflex, getup_s=4.0)
  assert not r["tool_on_fork"]
  assert r["stood"] is not None
  assert am.FOLD_ON_FALL_COS == pytest.approx(0.5)


# ---- the rack ------------------------------------------------------------------------

def test_the_racks_committed_tags_are_the_generators():
  from PIL import Image
  from pluggybot.rack.tags import TAG_DIR, tag_image
  for tag_id in rk.RACK_TAG_IDS:
    png = np.asarray(Image.open(ROOT / TAG_DIR / f"tag{tag_id}.png"))
    assert np.array_equal(png, tag_image(tag_id))
  assert len(rk.RACK_TAG_IDS) == len(rk.DEFAULT.tag_ys)


def test_a_bays_two_tags_stay_in_the_nose_cameras_view_from_the_working_pose():
  # From the working pose, turned as far as the line-up allows and a degree
  # more, both of a bay's tags are inside the nose camera's view. Between
  # the bays (+-150 mm) they were 25 deg off its axis, and 2.7 deg of yaw
  # took one out: a fetch at the bay had one tag and no fit.
  model, data = _body()
  cam = model.camera("nav_eye").id
  half = math.atan(math.tan(math.radians(model.cam_fovy[cam] / 2)) * 16 / 9)
  cam_x = data.cam_xpos[cam][0] - data.xpos[1][0]
  depth = rk.WORK_X - (rk.DEFAULT.back_x + 0.006) - cam_x
  from pluggybot.rack.tags import plate_half_extent
  edge = plate_half_extent(rk.DEFAULT.tag_size)

  def worst(dy):
    yaw = rk.LINEUP_YAW + math.radians(1.0)
    return math.atan2(dy + edge, depth) + yaw
  assert worst(rk.DEFAULT.tag_dy) < half
  assert worst(0.15) > half


def test_the_rack_fit_needs_a_baseline_and_refuses_a_misplaced_tag():
  pose = (0.8, -0.05, math.pi - 0.05)
  layout = rk.tag_layout()

  def at(p, x, y):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (p[0] + c * x - s * y, p[1] + s * x + c * y)
  seen = {i: at(pose, x, y) for i, (x, y, _) in layout.items()}
  fix = rk.fit_rack(seen)
  assert (fix.x, fix.y) == pytest.approx(pose[:2], abs=1e-9)
  assert math.cos(fix.yaw - pose[2]) == pytest.approx(1.0)
  assert rk.fit_rack({29: seen[29]}) is None
  # One tag of three 80 mm off where the drawing puts it is refused (the
  # fit spreads it, so 60 mm reads 28 mm rms and passes).
  bad = {i: seen[i] for i in (30, 31, 32)}
  bad[31] = (bad[31][0] + 0.08, bad[31][1])
  assert rk.fit_rack(bad) is None
  bad[31] = (bad[31][0] - 0.08 + 0.005, bad[31][1])       # 5 mm: a real look's
  assert rk.fit_rack(bad) is not None


def test_a_bays_aim_is_its_peg_where_it_crosses_the_arms_plane():
  # Synthetic tags from a rack 3 deg off square and 12 mm across: the aim is
  # the peg's axis where it meets the arm's plane, at the peg's height.
  spec = rk.DEFAULT
  yaw = math.pi + math.radians(3.0)
  bay = 1
  bx, by = 0.45, 0.012 - spec.bays[bay] * math.cos(yaw)
  c, s = math.cos(yaw), math.sin(yaw)
  seen = {i: np.array([bx + c * x - s * y, by + s * x + c * y, z - 0.35])
          for i, (x, y, z) in rk.tag_layout(spec).items()}
  aim = rk.bay_aim(seen, spec, bay)
  peg = (bx - s * spec.bays[bay], by + c * spec.bays[bay])
  assert aim.across == pytest.approx(peg[1], abs=1e-9)
  assert aim.yaw == pytest.approx(math.radians(3.0), abs=1e-9)
  assert aim.x == pytest.approx(peg[0] + peg[1] * math.tan(math.radians(3.0)), abs=1e-6)
  assert aim.z == pytest.approx(spec.peg_z - 0.35, abs=1e-9)


def test_the_walk_in_never_asks_the_policy_to_creep():
  # The flat policy does not walk below ~0.2 m/s or turn below ~0.2 rad/s
  # (the dock's finding): every command is a stop, a turn at TURN_W, or a
  # walk at APPROACH_V.
  from pluggybot.legs.policy import Twist
  for ex in np.linspace(-1.3, 0.05, 28):
    for ey in np.linspace(-0.3, 0.3, 13):
      for eth in np.radians(np.linspace(-40, 40, 17)):
        tw = rk.walk_in_twist(ex, ey, eth)
        if tw == Twist():
          continue
        if tw.vx == 0.0:
          assert abs(tw.yaw_rate) == rk.TURN_W
        else:
          assert tw.vx == rk.APPROACH_V
  assert rk.walk_in_twist(-rk.STOP_M, 0.0, 0.0) == Twist()
  assert rk.walk_in_twist(-rk.STOP_M - 0.01, 0.0, 0.0).vx > 0
