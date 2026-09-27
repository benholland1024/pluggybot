"""The quadruped's dock (issue #378): a cradle the robot lies down onto.
SimNotes, "The quadruped's dock", has the tables; `scripts/dock_spike.py`
flies them. These pin the rules the tables settled, each as cheaply as it
can still fail for the right reason."""

from dataclasses import replace
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from pluggybot.legs import dock as dk
from pluggybot.legs import model as qm
from pluggybot.legs.drivers import Drivers
from pluggybot.legs.policy import Twist

ROOT = Path(__file__).resolve().parents[1]


def _world(spec=dk.DEFAULT):
  model = mujoco.MjModel.from_xml_string(*dk.world_xml(spec))
  return model, mujoco.MjData(model)


def _lying(spec=dk.DEFAULT, *, y=0.0, yaw=0.0, lift=0.002, seconds=0.5):
  """The robot in its rest posture above the seat, let go with the drivers
  holding nothing, `seconds` later."""
  model, data = _world(spec)
  mujoco.mj_resetDataKeyframe(model, data, 1)       # "lie"
  data.qpos[1] = y
  data.qpos[2] = spec.bed_z + qm.CHOSEN.belly_depth + lift
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(model, data)
  Drivers(model, data).limp()
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
  return model, data


# ---- the criterion ------------------------------------------------------------

def test_a_robot_lying_on_the_dock_is_charging_and_one_lying_reversed_is_not():
  # Each pad must meet a pin of ITS OWN pole: turned round, the robot's left
  # pad lies on the right pole, which is a short, not a charge.
  model, data = _lying()
  assert dk.dock_charge_contact(model, data)
  model, data = _lying(yaw=math.pi)
  assert not dk.dock_charge_contact(model, data)
  # ...and turned round it still lies seated: the refusal is the pairing.
  assert abs(data.qpos[2] - (dk.DEFAULT.bed_z + qm.CHOSEN.belly_depth)) < 0.002


def test_the_pins_press_with_their_springs_whatever_the_weight_does():
  # Lying on the bed puts each pin at its rated travel: the pole presses
  # its pad with the datasheet's force, two pins' worth. Rigid contacts
  # under a belly and four limp legs carried whatever the legs left them,
  # and a robot lying 6 mm off centre lifted one pad clean off.
  model, data = _lying(seconds=1.0)
  k, ref = dk.pole_spring()
  for lbl in ("l", "r"):
    q = data.qpos[model.jnt_qposadr[model.joint(f"dock_pole_{lbl}").id]]
    assert abs(q + dk.PIN_TRAVEL) < 0.0003          # at mid-stroke
    force = k * (ref - q)
    assert abs(force - dk.PINS_PER_POLE * dk.PIN_MID_N) < 0.35


def test_the_criterion_holds_every_step_of_lying_there():
  # A pole whose plungers weighed a gram chattered against the stiff pin
  # contact: the criterion flipped every few steps, so a charge would have
  # run at the fraction of steps it happened to read true.
  model, data = _lying(y=0.004, seconds=0.5)
  held = 0
  for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)                       # still limp
    held += dk.dock_charge_contact(model, data)
  assert held == int(1.0 / model.opt.timestep)


# ---- the funnel -------------------------------------------------------------------

@pytest.mark.parametrize("cradle_mu, charges", [(dk.DEFAULT.cradle_mu, True), (1.0, False)])
def test_the_funnel_centres_a_belly_only_while_its_faces_are_slippery(cradle_mu, charges):
  # Let go 30 mm up, 15 mm off the axis: the belly lands on a face and
  # slides into the bed. At the belly case's own friction (the pair's MAX
  # without priority, 1.0) the face holds it where it landed, pads off the
  # pins. The funnel is only a funnel while tan(face angle) beats the
  # friction with room to spare.
  spec = replace(dk.DEFAULT, cradle_mu=cradle_mu)
  model, data = _lying(spec, y=0.015, lift=0.030, seconds=1.0)
  assert dk.dock_charge_contact(model, data) is charges


def test_a_belly_let_go_above_the_dock_lands_on_it_not_through_it():
  # MuJoCo's soft contact let a 9 kg belly falling 30 mm sink through the
  # 6 mm plate onto the floor; the plate's collision box is keeled under it.
  model, data = _lying(lift=0.030, seconds=1.0)
  assert abs(data.qpos[2] - (dk.DEFAULT.bed_z + qm.CHOSEN.belly_depth)) < 0.002
  assert dk.dock_charge_contact(model, data)


def test_the_dock_leaves_the_feet_room_walking_straight():
  # The walk-in steers onto the axis before its front feet reach the dock
  # and walks straight over it (`walk_in_twist`), where the nearest stance
  # foot's centre is FOOT_TRACK_M off the centreline. A wider dock, or a
  # body with a narrower stance, puts a foot on the funnel's face.
  inner_edge = dk.FOOT_TRACK_M - qm.CHOSEN.foot_r
  assert dk.DEFAULT.outer_half < inner_edge - 0.005
  # The funnel's mouth is still wider than the belly: it captures.
  model, _ = _world()
  belly = model.geom_size[model.geom("belly").id][1]
  assert dk.DEFAULT.mouth_half - belly >= 0.02


# ---- the board ----------------------------------------------------------------------

def test_the_boards_committed_tags_are_the_generators():
  from PIL import Image
  from pluggybot.rack.tags import DOCK_TAG_IDS, TAG_DIR, tag_image
  for tag_id in DOCK_TAG_IDS:
    png = np.asarray(Image.open(ROOT / TAG_DIR / f"tag{tag_id}.png"))
    assert np.array_equal(png, tag_image(tag_id)), (
      "stale: uv run python -m pluggybot.legs.dock")


def test_the_fit_takes_the_facing_from_the_baseline_between_tags():
  import itertools
  pose = (1.3, -0.12, math.radians(12.0))
  layout = dk.tag_layout()
  seen = {i: dk.compose(pose, (x, y, 0.0))[:2] for i, (x, y, _) in layout.items()}
  fix = dk.fit_dock(seen)
  assert (fix.x, fix.y, fix.yaw) == pytest.approx(pose, abs=1e-9)
  # Any two across the board do -- a row's pair, a diagonal -- but a
  # column's two tags are ONE point in the plane: no baseline, no facing
  # (seen alone they fitted any heading as 0, with an rms of 0).
  for a, b in itertools.combinations(seen, 2):
    two = dk.fit_dock({a: seen[a], b: seen[b]})
    if layout[a][:2] == layout[b][:2]:
      assert two is None
    else:
      assert two.yaw == pytest.approx(pose[2], abs=1e-9)
  assert dk.fit_dock({25: seen[25]}) is None


def test_a_tag_that_is_not_where_the_drawing_says_refuses_the_fit():
  # A misdecoded tag or a range gone wrong is refused, not blended in: the
  # walk-in folds each look half into what it believes.
  pose = (1.0, 0.0, 0.0)
  seen = {i: dk.compose(pose, (x, y, 0.0))[:2] for i, (x, y, _) in dk.tag_layout().items()}
  seen[25] = (seen[25][0] + 0.1, seen[25][1])
  assert dk.fit_dock(seen) is None
  seen[25] = (seen[25][0] - 0.1 + 0.005, seen[25][1])     # 5 mm: a real look's
  assert dk.fit_dock(seen) is not None


def test_the_boards_baseline_keeps_a_depth_error_off_the_seat():
  # The facing is fitted to where the tags are, and the seat is board_x
  # behind them, so ranges that disagree left to right turn the believed
  # axis and swing the seat sideways. From 1.6 m the left and right tags'
  # PnP ranges disagreed by 3-4 mm; walking in with the tags +-70 mm apart
  # the robot believed itself 17 mm off an axis it walked within 7 mm of.
  truth = (1.0, 0.0, 0.0)
  seen = {i: dk.compose(truth, (x, y, 0.0))[:2] for i, (x, y, _) in dk.tag_layout().items()}
  split = {i: (p[0] + (0.002 if p[1] > 0 else -0.002), p[1]) for i, p in seen.items()}
  assert abs(dk.fit_dock(split).y - truth[1]) < 0.010


@pytest.mark.parametrize("stance", ["standing to lie", "lying"])
def test_the_robot_reads_its_pose_off_the_board_standing_to_lie_and_lying(stance):
  # Standing where it lies down from, the camera on its nose sees the
  # board's upper pair; lying on the seat, the lower pair. Either gives its
  # pose in the dock's frame -- the anchor -- to millimetres.
  from pluggybot.rack.tags import DOCK_TAG_SIZE, TagDetector
  from pluggybot.telemetry.protocol import ROBOT_ROOT
  if stance == "lying":
    model, data = _lying()
  else:
    model, data = _world()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0] = dk.LIE_SHIFT_M
    mujoco.mj_forward(model, data)
  root = model.body(ROBOT_ROOT).id
  det = TagDetector(model, "nav_eye", tag_size=DOCK_TAG_SIZE)
  try:
    seen = dk.seen_from(model, data, det.detect(data), "nav_eye", root)
  finally:
    det.close()
  fix = dk.fit_dock(seen)
  assert fix is not None and fix.n >= 2
  x, y, th = dk.relative((0.0, 0.0, 0.0), (fix.x, fix.y, fix.yaw))
  w, qx, qy, qz = data.qpos[3:7]
  true_yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
  assert math.hypot(x - data.qpos[0], y - data.qpos[1]) < 0.006
  assert abs(th - true_yaw) < math.radians(0.5)


# ---- the walk in ------------------------------------------------------------------

def test_the_walk_in_never_asks_the_policy_to_creep():
  # The flat policy does not walk below ~0.2 m/s or turn below ~0.2 rad/s
  # (0.15 m/s commanded walked 5 mm/s): every command is a stop, a turn on
  # the spot at TURN_W, or a walk at APPROACH_V.
  for ex in np.linspace(-1.3, 0.1, 29):
    for ey in np.linspace(-0.3, 0.3, 13):
      for eth in np.radians(np.linspace(-40, 40, 17)):
        tw = dk.walk_in_twist(ex, ey, eth)
        if tw == Twist():
          continue
        if tw.vx == 0.0:
          assert abs(tw.yaw_rate) == dk.TURN_W and tw.vy == 0.0
        else:
          assert tw.vx == dk.APPROACH_V
        if ex > dk.over_dock_x():
          # Over the dock: straight, lined up, no sidestep, no turn on the
          # spot -- or it stops for the check to back it out.
          assert tw.vy == 0.0 and abs(tw.yaw_rate) <= dk.OVER_YAW_MAX
          assert tw.vx == dk.APPROACH_V and abs(ey) <= dk.OVER_ACROSS_M
  # It stops where it lies down from, not at the seat.
  assert dk.walk_in_twist(dk.LIE_SHIFT_M - dk.STOP_M, 0.0, 0.0) == Twist()
  assert dk.walk_in_twist(dk.LIE_SHIFT_M - dk.STOP_M - 0.01, 0.0, 0.0).vx > 0


def test_the_flat_policy_stops_where_the_walk_in_cuts_its_command():
  # STOP_M is the committed policy's coast from APPROACH_V: a retrained
  # policy that stops elsewhere lands every docking that far off.
  from pluggybot.legs.policy import PolicyDriver, WalkingPolicy
  model = mujoco.MjModel.from_xml_string(qm.body_xml(qm.CHOSEN))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  drv = PolicyDriver(model, data, WalkingPolicy())
  for _ in range(int(1.5 / model.opt.timestep)):
    drv.step(Twist(vx=dk.APPROACH_V))
  x0 = data.qpos[0]
  for _ in range(int(1.5 / model.opt.timestep)):
    drv.step(Twist())
  assert abs((data.qpos[0] - x0) - dk.STOP_M) < 0.015


def test_standing_lie_shift_ahead_of_the_seat_it_lies_down_onto_it():
  # #377's scripted lie-down folds the legs forward under the belly and the
  # body travels back, so the walk-in stops LIE_SHIFT_M ahead of the seat.
  # A changed lie-down, or a bed the belly slides on differently, lands
  # every docking that much off the seat.
  sys.path.insert(0, str(ROOT / "scripts"))
  import quad_spike as qs
  from pluggybot.legs.policy import PolicyDriver, WalkingPolicy
  from pluggybot.legs.scripted import VirtualModel
  model, data = _world()
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qpos[0] = dk.LIE_SHIFT_M
  mujoco.mj_forward(model, data)
  # It lies down from the stance the policy stopped it in.
  drv = PolicyDriver(model, data, WalkingPolicy())
  for _ in range(int(0.5 / model.opt.timestep)):
    drv.step(Twist())
  x0 = data.qpos[0]
  for tau in qs.lie_down_routine(qm.CHOSEN, data, VirtualModel(model, data, qm.CHOSEN)):
    drv.drivers.torque(tau)                          # the routine's torque
    mujoco.mj_step(model, data)
  assert abs((x0 - data.qpos[0]) - dk.LIE_SHIFT_M) < 0.003
  assert dk.dock_charge_contact(model, data)


def test_the_dock_is_one_body_with_two_sprung_poles_and_no_other_joint():
  # A world carrying the dock gains exactly its two pin poles: the rest of
  # it is furniture.
  model, _ = _world()
  bare = mujoco.MjModel.from_xml_string(qm.body_xml(qm.CHOSEN))
  assert model.njnt - bare.njnt == 2
  for lbl in ("l", "r"):
    j = model.joint(f"dock_pole_{lbl}")
    assert j.type[0] == mujoco.mjtJoint.mjJNT_SLIDE
    assert tuple(j.range) == pytest.approx((-dk.PIN_STROKE, 0.0))
