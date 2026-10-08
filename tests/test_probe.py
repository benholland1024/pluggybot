"""The probe from the robot's own senses (issue #466, stage 2; `legs/probe.py`):
the robot's guess of the box off the D435's depth, the re-find lying, the
take clear of the bracket, the record it makes and the fence round it.

The flown claim -- the probe walks in, takes hold and sweeps the lid over
seeded set-outs, and how often it gets through -- is the batch's
(`scripts/probe_chest.py`, SimNotes "The probe from the robot's own
senses"). Every rule it stands on is pinned here without a flight: the depth
frames are the real camera's rays from the robot's real CAD, posed and not
stepped, in a world of a floor and the chest.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot.activity import chest as ch
from pluggybot.legs import probe as pr
from pluggybot.perception import box

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pluggybot"
FLOOR = ('<mujoco><compiler angle="radian" autolimits="true"/>'
         '<option timestep="0.002" integrator="implicitfast"/>'
         '<worldbody><geom name="floor" type="plane" size="20 20 0.1"/></worldbody></mujoco>')
#: A chest turned off the map's axes, so nothing lines up by accident.
CHEST = (1.2, 0.3, 0.35)
#: Its drop handle's turn on its pin at rest, rad (measured in the house).
REST_SWING = -0.653


@pytest.fixture(scope="module")
def scene():
  """The robot's CAD and the chest on a floor, and the D435 at the probe's
  resolution: `look(x, y, yaw, lying)` poses the robot (no physics) and
  returns the frame's points in the map, laid through the truth -- the
  belief and the IMU exact."""
  from pluggybot.legs import world as lw
  from pluggybot.legs.model import CHOSEN, attachable, lie_qpos
  from pluggybot.perception.depth import DepthCamera
  from pluggybot.robot import FIRST
  spec = mujoco.MjSpec.from_string(FLOOR)
  spec.attach(attachable(CHOSEN), prefix="", frame=spec.worldbody.add_frame())
  ch.attach_chest(spec, ch.Lid(), CHEST[:2], CHEST[2], tags=False)
  m = spec.compile()
  d = mujoco.MjData(m)
  cam = DepthCamera(m, FIRST, width=pr.DEPTH_W, height=pr.DEPTH_H, min_z=pr.DEPTH_MIN_Z,
                    mount="body", seed=pr.DEPTH_SEED)
  root = m.body("pluggybot").id
  q0 = int(m.jnt_qposadr[m.body_jntadr[root]])
  # the handle as it rests: its centre of mass under its pin, swung 37.4 deg
  d.qpos[m.jnt_qposadr[m.joint("chest_pin").id]] = REST_SWING

  def look(x, y, yaw, lying=False, frames=3):
    lw.stand(m, d, "", x, y, yaw)
    if lying:
      d.qpos[q0 + 2] = CHOSEN.belly_depth
      j = int(m.jnt_qposadr[m.joint("FL_hip_abd").id])
      d.qpos[j:j + 12] = lie_qpos(CHOSEN)
      mujoco.mj_forward(m, d)
    rot = d.xmat[root].reshape(3, 3)
    return np.concatenate([cam.frame(d).points @ rot.T + d.xpos[root] for _ in range(frames)])

  mujoco.mj_forward(m, d)
  truth = {"knob": d.geom_xpos[m.geom("chest_knob").id].copy(),
           "pin": d.xanchor[m.joint("chest_pin").id].copy(),
           "hinge": d.xanchor[m.joint("chest_hinge").id].copy()}
  return SimpleNamespace(look=look, **truth)


def _u() -> np.ndarray:
  return np.array([math.cos(CHEST[2]), math.sin(CHEST[2]), 0.0])


def _stance(scene, off: float) -> tuple[float, float, float]:
  x, y = scene.knob[:2] - off * _u()[:2]
  return float(x), float(y), CHEST[2]


@pytest.fixture(scope="module")
def measured(scene):
  """The guess from the measuring stance, and the cloud it was read off."""
  cloud = scene.look(*_stance(scene, pr.MEASURE_AT_M))
  return pr.guess(cloud, scene.knob, CHEST[2] + 0.2), cloud


# ---- the guess: code's foresight, measured off depth ---------------------------------

def test_the_guess_finds_the_hinge_and_the_pin_off_depth(scene, measured):
  # The hinge is the top's back edge at the top's height -- a board over the
  # true axis, the foresight's stated cost -- and the pin the tip of the
  # bracket's top, half the bracket over the pin.
  g, _ = measured
  assert isinstance(g, pr.Guess), g
  u = _u()
  along = lambda p, q: float((np.subtract(p, q)) @ u)  # noqa: E731
  assert abs(along(g.hinge, scene.hinge)) < 0.002
  assert g.hinge[2] - scene.hinge[2] == pytest.approx(ch.LID_T, abs=0.0015)
  assert abs(along(g.pin, scene.pin)) < 0.002
  assert g.pin[2] - scene.pin[2] == pytest.approx(0.004, abs=0.0015)
  assert g.yaw == pytest.approx(CHEST[2], abs=math.radians(0.3))
  r = scene.pin - scene.hinge
  a = np.array([-math.sin(CHEST[2]), math.cos(CHEST[2]), 0.0])
  assert g.radius == pytest.approx(float(np.linalg.norm(r - (r @ a) * a)), abs=0.003)


def test_the_bracket_tip_is_read_with_its_face_toward_the_camera(scene, measured, monkeypatch):
  # Its front face stands its top at the tip: read off the mean of a window
  # (as a top's far edge is), the tip came 4 mm past it.
  _, cloud = measured
  monkeypatch.setattr(pr.box, "faced_edge", lambda along, far=True: box.edge(along, far=far))
  g = pr.guess(cloud, scene.knob, CHEST[2] + 0.2)
  assert float(np.subtract(g.pin, scene.pin) @ _u()) < -0.003


def test_a_cloud_of_no_box_is_said_so():
  floor = np.column_stack([np.random.default_rng(0).uniform(0, 1, (500, 2)), np.zeros(500)])
  assert pr.guess(floor, (0.5, 0.5, 0.1), 0.0) == "no top seen"


# ---- the path, the take -----------------------------------------------------------

def test_a_held_knob_moves_as_its_pin_does_never_as_a_point_on_the_lid():
  # The knob hangs on a drop handle; held, the handle keeps its turn and the
  # knob rides the pin's arc. Planned as a point fixed to the lid -- the
  # knob's own arc about the hinge -- it is 66 mm off at the top, past the
  # 40 mm Kp 8 forgives (#469).
  hinge = ch.HINGE
  pin = ch.HINGE + ch.PIN
  knob = pin + ch.KNOB_OFF
  axis = (0.0, 1.0, 0.0)
  path = pr.pin_path(knob, pin, hinge, axis)
  assert np.allclose(path(0.0), knob)
  for s in (0.3, 0.8, pr.TOP):
    turned_pin = hinge + pr.rot_about(axis, s) @ (pin - hinge)
    assert np.allclose(path(s) - turned_pin, knob - pin), "the knob does not turn"
  fixed = hinge + pr.rot_about(axis, pr.TOP) @ (knob - hinge)
  assert np.linalg.norm(fixed - path(pr.TOP)) > 0.06


def test_a_knob_hanging_under_its_bracket_is_taken_clear_of_it():
  # A drop handle rests with its centre of mass under its pin, its knob
  # 9.6 mm in front of the bracket's tip: straight down onto it the claw's
  # crossbar landed on the bracket 42 mm over the knob.
  m = mujoco.MjModel.from_xml_string(ch.chest_xml(ch.Lid(), tags=False).replace(
    "<worldbody>", '<option timestep="0.002"/><worldbody>', 1))
  d = mujoco.MjData(m)
  for _ in range(4000):                                   # 8 s to rest
    mujoco.mj_step(m, d)
  ahead = float(d.xanchor[m.joint("chest_pin").id][0] - d.geom_xpos[m.geom("chest_knob").id][0])
  assert 0.0 < ahead < pr.TAKE_CLEAR_M, "the premise: the knob hangs under the crossbar's reach"
  frame = pr.Frame(0.0, 0.0, 0.0, 0.0)
  g = pr.Guess(knob=(0.5, 0.0, 0.09), yaw=0.0, top=0.152, hinge=(0.83, 0.0, 0.152),
               pin=(0.5 + ahead, 0.0, 0.15))
  assert pr.Probe.take_point(g, frame)[0] == pytest.approx(g.pin[0] - pr.TAKE_CLEAR_M)
  far = pr.Guess(knob=(0.5, 0.0, 0.09), yaw=0.0, top=0.152, hinge=(0.83, 0.0, 0.152),
                 pin=(0.53, 0.0, 0.15))
  assert np.allclose(pr.Probe.take_point(far, frame), (0.5, 0.0, 0.09))


# ---- lying: the box re-found ----------------------------------------------------------

def _lay(cloud: np.ndarray, error) -> np.ndarray:
  """A cloud as a robot whose belief is off the truth by `error` (along x,
  along y, a turn -- each of the map's) lays it: the world turned and moved
  by it about the map's origin."""
  dx, dy, dth = error
  c, s = math.cos(dth), math.sin(dth)
  out = cloud.copy()
  out[:, 0] = c * cloud[:, 0] - s * cloud[:, 1] + dx
  out[:, 1] = s * cloud[:, 0] + c * cloud[:, 1] + dy
  return out


def test_lying_the_box_is_refound_by_its_front_face_and_the_guess_moved_with_it(scene, measured):
  # Lying, the D435 sees the box's front face and nothing over it; between
  # the measure and lying the belief walked up to 25 mm (36 mm off the
  # truth once, and the jaws shut on the handle). The guess is moved by what
  # the face says the belief did.
  g, standing = measured
  x, y, yaw = _stance(scene, pr.KNOB_AHEAD_M)
  lying = scene.look(x, y, yaw, lying=True, frames=pr.LYING_FRAMES)
  before = pr.front_face(standing, g)
  same = pr.front_face(lying, g)
  assert before is not None and same is not None
  assert same[0] == pytest.approx(before[0], abs=0.001), "two looks, one face"
  assert same[1] == pytest.approx(before[1], abs=0.003)
  # the face's place along: the chest's front, off the knob along the box
  assert before[0] == pytest.approx(float((np.array(CHEST[:2]) - g.knob[:2]) @ _u()[:2]),
                                    abs=0.001)
  error = (0.02, -0.012, math.radians(0.6))
  after = pr.front_face(_lay(lying, error), g)
  moved = pr.Move(g, before, after).guess()
  for p in ("knob", "pin", "hinge"):
    want = _lay(np.array([getattr(g, p)]), error)[0]
    assert np.allclose(getattr(moved, p)[:2], want[:2], atol=0.0015), p
    assert getattr(moved, p)[2] == getattr(g, p)[2]
  assert moved.yaw == pytest.approx(g.yaw + error[2], abs=math.radians(0.15))


def _in_box(p) -> np.ndarray:
  """A map point in the chest's own frame, mm."""
  c, s = math.cos(CHEST[2]), math.sin(CHEST[2])
  dx, dy = p[0] - CHEST[0], p[1] - CHEST[1]
  return 1000 * np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])


def test_the_box_is_measured_in_its_own_frame_off_its_front_face(scene, measured):
  # What its model's author is shown (#466 stage 3): the top's depth, width
  # and height, the bracket and the knob, in a frame on the floor under the
  # middle of its front face -- the chest's own, here.
  from pluggybot.imagination.record import Record
  g, cloud = measured
  rec = Record(start=_start(), dt=0.002, commands=np.zeros((1, 6)), depth=(cloud,))
  s = pr.sizes(rec, g)
  assert isinstance(s, pr.Sizes), s
  # the back edge within the 2 mm the hinge's own test allows, at a turn
  assert (s.depth, s.width, s.top) == (pytest.approx(1000 * ch.BOX_D, abs=2.0),
                                       pytest.approx(1000 * ch.BOX_W, abs=1.5),
                                       pytest.approx(1000 * (ch.BOX_H + ch.LID_T), abs=1.0))
  tip = ch.HINGE[0] + ch.PIN[0]                            # the bracket ends at the pin
  assert s.tip == pytest.approx(1000 * tip, abs=1.0)
  assert s.bracket_width == pytest.approx(12.0, abs=1.0)
  assert s.bracket_top == pytest.approx(1000 * (ch.BOX_H + ch.PIN_Z + 0.004), abs=1.0)
  assert np.allclose(s.origin, 1000 * np.array(CHEST[:2]), atol=1.0)
  assert s.yaw == pytest.approx(math.degrees(CHEST[2]), abs=0.3)
  assert np.allclose(s.knob, _in_box(scene.knob), atol=1.0)
  assert s.knob_size == 26.0


def test_the_front_is_its_faces_place_never_the_tops_near_edge(scene, measured):
  # The premise: the face's own points stand at the top's near edge and pull
  # a window's mean toward the robot (`perception.box`'s faced edge) -- off
  # the top, the front read 7.5 mm short and the box 227 mm deep.
  g, cloud = measured
  u = np.array([math.cos(g.yaw), math.sin(g.yaw)])
  v = np.array([-u[1], u[0]])
  rel = cloud[:, :2] - np.asarray(g.knob[:2])
  pu, pv = rel @ u, rel @ v
  on = box.on_top(cloud, g.top)
  near = box.edge(pu[on & (np.abs(pv) > pr.BESIDE_M)], far=False)
  face = pr.front_face(cloud, g)[0]
  truth = float((np.array(CHEST[:2]) - g.knob[:2]) @ u)
  assert face == pytest.approx(truth, abs=0.001)
  assert truth - near > 0.004, "the top's near edge reads toward the robot"


def _start():
  from pluggybot.imagination.record import Start
  return Start(pose=(0.0, 0.0, 0.0), attitude=(0.0, 0.0), legs=(0.0,) * 12, arm=(2.2, 0.4))


def test_the_jaws_path_an_author_is_shown_is_the_encoders_in_the_boxs_frame():
  # Its motion as its arm felt it, in the frame the author draws in: a box
  # turned a quarter round, 1 m ahead, sees the jaws' x as its own -y.
  from pluggybot.imagination.record import Record
  from pluggybot.legs.model import CHOSEN
  n = 1000
  sensed = np.zeros((n, 4))
  sensed[:, 2], sensed[:, 3] = 2.2, np.linspace(0.4, 0.9, n)
  rec = Record(start=_start(), dt=0.002, commands=np.zeros((n, 6)), sensed=sensed)
  frame = pr.Frame(0.0, 0.0, 0.0, 0.105)
  probed = pr.Probed(record=rec, frame=frame, phases={"take": (0, 200), "up0": (200, n)})
  s = pr.Sizes(origin=(1000.0, 0.0), yaw=90.0, depth=220.0, width=300.0, top=152.0,
               tip=-50.0, bracket_top=150.0, bracket_width=12.0, knob=(-60.0, 0.0, 88.0),
               knob_size=26.0)
  lines = pr.did(probed, s).splitlines()
  points = [ln for ln in lines if ln.strip().endswith(")")]
  assert len(points) == pr.PATH_POINTS and points[0].split()[0] == "0.0"
  jaws = 1000 * pr.jaws_seen(rec, frame, CHOSEN.arm)
  import re
  x, y, z = (float(t) for t in re.findall(r"-?\d+\.\d", points[-1].split("s", 1)[1]))
  assert (x, y, z) == (pytest.approx(jaws[n - 1, 1], abs=0.1),
                       pytest.approx(1000.0 - jaws[n - 1, 0], abs=0.1),
                       pytest.approx(jaws[n - 1, 2], abs=0.1))


def test_the_calibration_sweeps_only_over_bare_floor():
  frame = pr.Frame(0.0, 0.0, 0.0, 0.3)
  cup = np.array([[0.5, 0.01, 0.06]])
  probe = object.__new__(pr.Probe)
  assert not probe.bare(cup, frame, 0.4, 0.6)
  assert probe.bare(cup, frame, 0.6, 0.8), "beyond the arc's footprint"
  assert probe.bare(cup * [1, 1, 0.1], frame, 0.4, 0.6), "on the floor is the floor"


# ---- the record --------------------------------------------------------------------

@pytest.fixture
def lying_claw():
  """The robot lying in a world of its own with the claw on its fork (the
  imagination's: `imagination.compile.robot_spec`), and a mission of what
  the recorder reads."""
  from pluggybot.imagination.compile import Imagined, robot_spec
  from pluggybot.legs import arm as am
  from pluggybot.legs.imagined import ImaginedBody
  from pluggybot.legs.model import CHOSEN, lie_qpos
  from pluggybot.perception.imu import Attitude
  from pluggybot.robot import FIRST
  m = robot_spec("module_claw").compile()
  d = mujoco.MjData(m)
  body = ImaginedBody(m, d, "module_claw")
  qs, qe = am.CARRY_Q
  first = np.array([qs, qs + qe, pr.KP, pr.KD, 0.02, 0.0])
  body.place((0.0, 0.0, 0.0), (0.0, 0.0), lie_qpos(CHOSEN), (qs, qs + qe), first)
  for _ in range(500):                                    # come to rest
    body.apply(first)
    mujoco.mj_step(m, d)
  mis = SimpleNamespace(model=m, data=d, arm=body.arm, arm_spec=CHOSEN.arm, handle=FIRST,
                        step_hooks=[], pose=(0.0, 0.0, 0.0), carrying="module_claw",
                        posture="lying", falls=0,
                        odo=SimpleNamespace(att=Attitude(d.xquat[m.body("pluggybot").id])))
  world = Imagined(model=m, hooks=(), joints={}, carrying="module_claw")
  return SimpleNamespace(m=m, d=d, body=body, mis=mis, world=world, first=first)


def test_a_record_is_what_was_sent_and_sensed_and_replays_where_it_was_flown(lying_claw):
  # The fence's input: each row the arm drivers' targets AS THEY APPLIED
  # THEM -- ramped from a goal sent at once, as the live arm's are -- their
  # gains, the slide and the jaws; and after the step, the drivers' torque
  # readings and the encoders in their field. Replayed from its own start in
  # the robot's own world, it reads what was sensed.
  from pluggybot.imagination.rollout import rollout
  from pluggybot.legs import rack as rk
  from pluggybot.perception.encoders import torque_reading
  lc = lying_claw
  arm, d = lc.body.arm, lc.d
  slide = lc.m.actuator(rk.CLAW_SLIDE).id
  rec = pr.Recorder(lc.mis)
  rec.begin()
  q0 = arm.target.copy()
  arm.goal = q0 + [0.25, -0.05]                           # sent at once; the driver ramps it
  targets = []
  for k in range(400):
    d.ctrl[slide] = 0.02 - 0.01 * min(k, 200) / 200
    arm.step()
    targets.append(arm.target.copy())
    mujoco.mj_step(lc.m, d)
    for hook in lc.mis.step_hooks:
      hook()
  record = rec.record()
  assert rec._row not in lc.mis.step_hooks
  assert np.allclose(record.commands[:, :2], targets)
  assert not np.allclose(record.commands[:50, :2], arm.goal), "the premise: the goal ran ahead"
  assert np.allclose(record.commands[:, 2:4], (pr.KP, pr.KD))
  assert record.commands[-1, 4] == pytest.approx(0.01)
  step = int(round(d.time / lc.m.opt.timestep))
  act = arm.act[0]
  assert record.sensed[-1, 0] == torque_reading(float(d.actuator_force[act]), f"{act}", step)
  assert np.allclose(record.sensed[-1, 2:], arm.q(), atol=pr.POSITION_LSB)
  again = rollout(lc.world, record)
  assert np.abs(again.sensed[:, 2:] - record.sensed[:, 2:]).max() < 3 * pr.POSITION_LSB
  tau = again.sensed[:, :2] - record.sensed[:, :2]
  assert np.sqrt((tau ** 2).mean()) < 0.06 and abs(float(tau.mean())) < 0.01


def test_the_start_is_the_tilt_the_imagined_body_is_put_down_at():
  # `legs.imagined` turns by yaw, then pitch, then roll: the start reads
  # them back in that order off the IMU.
  from pluggybot.legs.imagined import _quat
  from pluggybot.perception.imu import Attitude
  for roll, pitch in ((0.1, -0.05), (-0.07, 0.12)):
    mis = SimpleNamespace(odo=SimpleNamespace(att=Attitude(_quat(roll, pitch, 0.8))))
    assert pr.attitude(mis) == pytest.approx((roll, pitch), abs=1e-9)


def test_the_jaws_are_where_the_arms_encoders_put_them(lying_claw):
  # Straight under the seated peg, the slide's command across: the robot's
  # own reading of where it holds the knob (and so the lid's angle). Across
  # and up it is the jaws'; ahead the claw leans on its peg, up to 4 deg --
  # 12 mm at the jaws -- and the reading takes the lean for the jaws.
  from pluggybot.imagination.record import Record
  from pluggybot.legs import rack as rk
  lc = lying_claw
  rec = pr.Recorder(lc.mis)
  rec.begin()
  for _ in range(250):
    lc.body.apply(lc.first)
    mujoco.mj_step(lc.m, lc.d)
    for hook in lc.mis.step_hooks:
      hook()
  record = rec.record()
  frame = pr.Frame(0.0, 0.0, 0.0, float(lc.d.xpos[lc.m.body("pluggybot").id][2]))
  one = Record(start=record.start, dt=record.dt, commands=record.commands[-1:],
               sensed=record.sensed[-1:])
  seen = pr.jaws_seen(one, frame, lc.mis.arm_spec)[0]
  grip = lc.d.site_xpos[lc.m.site(rk.CLAW_GRIP).id]
  assert abs(seen[1] - grip[1]) < 0.002 and abs(seen[2] - grip[2]) < 0.002
  assert abs(seen[0] - grip[0]) < 0.175 * math.sin(math.radians(5.0))


def test_a_probe_crosses_the_wire_whole():
  from pluggybot.imagination.record import Record, Start
  from pluggybot.imagination.worker import pack, unpack
  st = Start(pose=(1.0, 2.0, 0.3), attitude=(0.01, -0.02), legs=(0.1,) * 12, arm=(2.2, 0.4),
             carrying="module_claw")
  rec = Record(start=st, dt=0.002, commands=np.ones((3, 6)), sensed=np.zeros((3, 4)),
               depth=(np.ones((4, 3)),), detections=({"id": 53, "knob": [1.0, 2.0, 0.1]},))
  g = pr.Guess(knob=(1, 2, 0.09), yaw=0.3, top=0.15, hinge=(1.3, 2.1, 0.15),
               pin=(1.05, 2.0, 0.15), seen={"top": 9})
  out = pr.Probed(ok=True, why="probed", record=rec, phases={"take": (0, 3)},
                  calibration=Record(start=st, dt=0.002, commands=np.zeros((2, 6)),
                                     sensed=np.zeros((2, 4))),
                  friction=(0.1, 0.12), guess=g, frame=pr.Frame(1.0, 2.0, 0.3, 0.105), t0=4.0,
                  log={"tries": [{"walkIn": "stopped"}]},
                  picture=pr.jpeg(np.full((72, 128, 3), 90, dtype=np.uint8)))
  assert out.picture[:3] == b"\xff\xd8\xff", "a JPEG"
  back = pr.Probed.from_wire(*unpack(pack(*out.to_wire())))
  assert back.picture == out.picture
  assert back.guess == g and back.frame == out.frame and back.phases == out.phases
  assert np.array_equal(back.record.commands, rec.commands)
  assert np.array_equal(back.record.depth[0], rec.depth[0])
  assert back.record.detections == rec.detections and back.calibration.n == 2
  assert (back.ok, back.why, back.friction, back.t0, back.log) == (
    True, "probed", (0.1, 0.12), 4.0, out.log)


# ---- the fence -----------------------------------------------------------------------

def test_the_probe_reads_nothing_of_the_worlds_model_of_the_box():
  # The robot probes from its own senses: nothing it can load is the
  # chest's activity, our grading or the spike that chose the chest, and
  # its source names no element of the chest, judges no grip off the world
  # (`ClawHand.held` reads the bodies between the pads) and reads no truth
  # through the body's helpers (#479's review: the arm driver's `gravity`
  # reads `qpos`, `floor_below` and `_height` the world's heights).
  from test_imagination import _closure  # noqa: I001 -- tests/ is on sys.path
  closure = _closure(["pluggybot.legs.probe"])
  assert "pluggybot.legs.probe" in closure and "pluggybot.perception.box" in closure
  assert not [m for m in closure if m.startswith(("pluggybot.activity.chest",
                                                  "pluggybot.evaluation"))], sorted(closure)
  for name in ("legs/probe.py", "perception/box.py"):
    tree = ast.parse((SRC / name).read_text())
    words = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str)}
    assert not [w for w in words if "chest_" in w or "mechanism_spike" in w], name
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)}
    assert not calls & {"held", "held_tag", "true_pose", "gravity", "floor_below",
                        "_height"}, name


def test_the_record_weighs_the_arm_by_its_own_senses(lying_claw):
  # #479's review: the arm driver's gravity model reads the torso's attitude
  # and the joints off `qpos`, and the friction the probe calibrates and its
  # word that it held the lid are read off the record's. The record's is its
  # encoders' and its IMU's: the driver's model at the truth, and where the
  # IMU is off, the IMU's.
  from pluggybot.legs.imagined import _quat
  from pluggybot.perception.imu import Attitude
  lc = lying_claw
  arm = lc.body.arm
  assert arm.gravity_sensed(arm.q(), lc.mis.odo.att.level()) == pytest.approx(
    arm.gravity(), abs=1e-4)
  lc.mis.odo = SimpleNamespace(att=Attitude(_quat(0.0, 0.05, 0.0)))
  rec = pr.Recorder(lc.mis)
  rec.begin()
  arm.step()
  mujoco.mj_step(lc.m, lc.d)
  for hook in lc.mis.step_hooks:
    hook()
  rec.stop()
  sensed = arm.gravity_sensed(rec.sensed[0][2:], lc.mis.odo.att.level())
  assert rec.gravity[0] == pytest.approx(sensed, abs=1e-12)
  assert np.abs(rec.gravity[0] - arm.gravity()).max() > 0.01


def test_the_legs_say_how_high_they_hold_the_torso():
  # The probe lays its depth over the floor by its legs: their encoders,
  # levelled by its IMU (#479's review: `floor_below` read the torso's and
  # the feet's TRUE heights). Tilted, it still reads its own height.
  from pluggybot.legs import world as lw
  from pluggybot.legs.imagined import _quat
  from pluggybot.legs.model import CHOSEN, LEGS, attachable
  from pluggybot.legs.odometry import LegOdometry
  spec = mujoco.MjSpec.from_string(FLOOR)
  spec.attach(attachable(CHOSEN), prefix="", frame=spec.worldbody.add_frame())
  m = spec.compile()
  d = mujoco.MjData(m)
  lw.stand(m, d, "", 0.0, 0.0, 0.0)
  root = m.body("pluggybot").id
  q0 = int(m.jnt_qposadr[m.body_jntadr[root]])
  d.qpos[q0 + 3:q0 + 7] = _quat(0.02, 0.06, 0.3)
  mujoco.mj_forward(m, d)
  odo = LegOdometry(m, d)
  odo.step()
  feet = [m.site(f"{leg}_foot").id for leg in LEGS]
  true = float(d.xpos[root][2] - min(d.site_xpos[f][2] for f in feet)) + odo.foot_r
  assert odo.height() == pytest.approx(true, abs=0.0015)


class _Arm:
  """The arm driver as the probe's ways out use it."""

  def __init__(self) -> None:
    self.kp, self.kd, self.held = 60.0, 4.0, None

  def q(self):
    return (0.3, -0.2)

  def hold_at(self, shoulder, elbow):
    self.held = (shoulder, shoulder + elbow)

  def aim(self, shoulder, elbow):
    pass


class _Recorder:
  """A recorder that records nothing."""

  def __init__(self, mis) -> None:
    self.t0 = 0.0

  def begin(self):
    pass

  def mark(self, phase):
    pass

  def stop(self):
    pass

  def record(self, **kw):
    return None

  def phases(self):
    return {}


def _steps(name, returns=True):
  """A routine of two steps, the first its name."""
  yield name
  yield "."
  return returns


def _drive(routine):
  """A routine run to its end: what it returned."""
  try:
    while True:
      next(routine)
  except StopIteration as stop:
    return stop.value


def _stubbed(monkeypatch, fell=lambda: False, reach=True, sweep=True):
  """A probe on a stubbed body: its hand, its arm, every routine two steps."""
  monkeypatch.setattr(pr, "Recorder", _Recorder)
  arm = _Arm()
  mis = SimpleNamespace(arm=arm, pose=(0.0, 0.0, 0.0), posture="lying",
                        _drive_routine=lambda s, v, w: _steps("hold"),
                        _vertex_goal=lambda: (0.4, 0.4), rest_routine=lambda: _steps("rest"),
                        stand_routine=lambda: _steps("stand"),
                        odo=SimpleNamespace(att=SimpleNamespace(level=lambda: np.eye(3))))
  hand = SimpleNamespace(fell=fell, centre_routine=lambda: _steps("centre"),
                         jaws_routine=lambda closed, settle=0.3: _steps("shut" if closed else "open"),
                         to_routine=lambda *a, **kw: _steps("to", reach),
                         move_routine=lambda *a, **kw: _steps("move"))
  probe = object.__new__(pr.Probe)
  probe.mis, probe.clouds, probe.detections, probe.picture = mis, [], [], None
  probe.follow_routine = lambda *a: _steps("sweep", sweep)
  probe.held_down = lambda rec, friction: 1.0
  return probe, hand, arm


@pytest.mark.parametrize("end", ["thrown", "closed"])
def test_the_probe_puts_its_arms_gains_back_however_it_ends(monkeypatch, end):
  # #479's review: the let-go ran after the gains' `finally`, so a stop
  # thrown in as the jaws opened -- or a stand-up closing the routine it
  # landed in -- left the arm at Kp 8 for whatever ran next. Back on every
  # way out, and held where it IS: stiff, a step to the target the compliant
  # arm lagged would kick the lid.
  from pluggybot.tick import MissionAborted
  probe, hand, arm = _stubbed(monkeypatch)
  g = probe._probe_routine(hand, lambda s: np.zeros(3), np.eye(3), pr.Probed())
  opened = 0
  while opened < 2:                          # the take opens the jaws, then the let-go
    opened += next(g) == "open"
  assert (arm.kp, arm.kd) == (pr.KP, pr.KD)
  if end == "thrown":
    with pytest.raises(MissionAborted):
      g.throw(MissionAborted("the day's stop"))
  else:
    g.close()
  assert (arm.kp, arm.kd) == (60.0, 4.0) and arm.held == arm.q()


def test_a_take_the_body_fell_from_is_said_as_a_fall(monkeypatch):
  # #479's review: a take that failed was "the knob out of reach" before a
  # fall could be said.
  probe, hand, arm = _stubbed(monkeypatch, fell=lambda: True, reach=False)
  out = pr.Probed()
  _drive(probe._probe_routine(hand, lambda s: np.zeros(3), np.eye(3), out))
  assert (out.ok, out.why, out.log["swept"]) == (False, "fell", False)
  assert (arm.kp, arm.kd) == (60.0, 4.0)


@pytest.mark.parametrize("fell, why", [(False, "the arc left the arm's reach"), (True, "fell")])
def test_a_calibration_is_judged_before_the_stand_that_ends_it(monkeypatch, fell, why):
  # `fell` reads any new posture, and the let-go stands the body up: read
  # after it, an arc out of reach was a fall.
  state = {"fell": fell}
  probe, hand, arm = _stubbed(monkeypatch, fell=lambda: state["fell"], sweep=False)

  def stand():
    state["fell"] = True
    yield "stand"
  probe.mis.stand_routine = stand
  probe.bare = lambda *a: True
  probe.frame_now = lambda: pr.Frame(0.0, 0.0, 0.0, 0.0)
  probe._hand = lambda: hand
  probe.reaches = lambda *a: None
  g = pr.Guess(knob=(0.46, 0.0, 0.09), yaw=0.0, top=0.15, hinge=(0.75, 0.0, 0.15),
               pin=(0.45, 0.0, 0.12))
  log = {}
  assert _drive(probe.calibrate_routine(g, np.zeros((0, 3)), log)) is None
  assert log["why"] == why and (arm.kp, arm.kd) == (60.0, 4.0)


# ---- our grading ------------------------------------------------------------------------

def test_a_set_out_is_seeded_in_the_storeroom_and_its_knob_in_the_robots_view():
  from pluggybot.evaluation import probe as ep
  from pluggybot.home import world as home
  assert ep.set_out(3) == ep.set_out(3) and ep.set_out(3) != ep.set_out(4)
  with pytest.raises(ValueError, match="whole number"):
    ep.set_out(-1)
  # the knob drawn plumb, and where it rests swung on its pin: no further and
  # no wider than the 128 set-outs the probe found its tag from
  cp, sp = math.cos(REST_SWING), math.sin(REST_SWING)
  swing = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
  places = (((ch.HINGE + ch.PIN + ch.KNOB_OFF)[:2], ep.KNOB_FROM, ep.BEARING),
            ((ch.HINGE + ch.PIN + swing @ ch.KNOB_OFF)[:2], (0.75, 0.985), math.radians(10.2)))
  for k in range(128):
    so = ep.set_out(k)
    assert so.lid == ch.draw(k)
    x, y, heading = so.start
    c, s = math.cos(so.chest[2]), math.sin(so.chest[2])
    for at, (near, far), wide in places:
      knob = np.array(so.chest[:2]) + np.array([[c, -s], [s, c]]) @ at
      bearing = math.atan2(knob[1] - y, knob[0] - x) - heading
      assert near - 1e-9 <= math.dist(knob, (x, y)) <= far + 1e-9 and abs(bearing) <= wide + 1e-9
    corners = [np.array(so.chest[:2]) + np.array([[c, -s], [s, c]]) @ p
               for p in ((0, -ch.BOX_W / 2), (0, ch.BOX_W / 2), (ch.BOX_D, -ch.BOX_W / 2),
                         (ch.BOX_D, ch.BOX_W / 2))]
    for cx, cy in corners:
      assert home.LAB_X[0] + 0.5 < cx < home.LAB_X[1] - 0.5
      assert home.STORE_Y[0] + 0.5 < cy < home.STORE_Y[1] - 0.5


def test_the_guess_is_graded_in_its_own_robots_frame():
  # The probe measures and plans through one belief, and the record is laid
  # in one: the belief's own error is no error of the guess. A guess that is
  # the truth laid through a belief off by (20 mm, -8 mm, 0.7 deg) grades
  # perfect; the same guess against the true pose is not.
  from pluggybot.evaluation import probe as ep
  chest = (25.0, -3.0, 0.3)
  world = ep.truth(chest)
  true_pose = (24.1, -3.2, 0.25)
  belief = (24.12, -3.208, 0.25 + math.radians(0.7))
  laid = lambda p: np.array([*ep.in_map((p[0], p[1], 0.0), true_pose, belief)[:2], p[2]])  # noqa: E731
  knob = world["pin"] + np.array([-0.04 * math.cos(0.3), -0.04 * math.sin(0.3), -0.06])
  g = pr.Guess(knob=tuple(laid(knob)), yaw=ep.in_map((0, 0, chest[2]), true_pose, belief)[2],
               top=0.152, hinge=tuple(laid(world["hinge"])), pin=tuple(laid(world["pin"])))
  graded = ep.geometry(g, belief, true_pose, world, knob)
  for key in ("hingeAlongMm", "hingeUpMm", "pinAlongMm", "pinAcrossMm", "pinUpMm", "radiusMm"):
    assert graded[key] == pytest.approx(0.0, abs=0.01), key
  assert graded["knobMm"] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
  assert graded["facingDeg"] == pytest.approx(0.0, abs=1e-6)
  assert abs(ep.geometry(g, true_pose, true_pose, world, knob)["pinAlongMm"]) > 5.0


def probe_chest():
  """`scripts/probe_chest.py`, imported -- the environment it sets for its
  own BLAS left as it was."""
  import importlib.util
  import os
  from unittest import mock
  spec = importlib.util.spec_from_file_location("probe_chest", ROOT / "scripts" / "probe_chest.py")
  script = importlib.util.module_from_spec(spec)
  with mock.patch.dict(os.environ):
    spec.loader.exec_module(script)
  return script


def test_a_record_replays_through_worlds_whose_handle_starts_where_it_hung(monkeypatch):
  # The flight's handle had hung a minute when the record began, and the
  # world's chest compiles it plumb: a rollout's second of settling left it
  # 4.8 deg short and its knob 4.7 mm off the reference's (#479's review).
  # Both worlds a record replays through start where its handle hung.
  from pluggybot.evaluation import imagined as im
  from pluggybot.evaluation import probe as ep
  from pluggybot.imagination.rollout import Readings, settled
  worlds, n = [], 4

  def rollout(world, record):
    worlds.append(world)
    return Readings(t=np.zeros(n), sensed=np.zeros((n, 4)),
                    joints={"pin": np.zeros(n), "hinge": np.zeros(n)})
  monkeypatch.setattr(ep, "rollout", rollout)
  monkeypatch.setattr(ep.im, "compare", lambda truth, other, rows: [])
  ep.replay(SimpleNamespace(n=n, dt=0.002, sensed=np.zeros((n, 4))), {"take": (0, n)},
            ch.draw(0), (0.30, 0.0, 0.0), REST_SWING, np.zeros(n))
  world, reference = worlds
  knobs = []
  for w, knob in ((world, lambda m, d: d.geom_xpos[m.geom("chest_knob").id]),
                  (reference, lambda m, d: d.xpos[m.body("scene_knob").id])):
    d = settled(w, im.hold_record())
    knobs.append(knob(w.model, d).copy())
  assert d.qpos[reference.joints["pin"][0]] == pytest.approx(0.0, abs=1e-3)
  assert np.linalg.norm(knobs[0] - knobs[1]) < 0.001, knobs


def test_a_handle_is_judged_by_its_turn_while_the_catch_held_and_never_after():
  # The flag (Ben, #479): shut, the handle's angle on its pin is its angle
  # in the world. Once the lid is past the catch's reach a drop handle turns
  # with the lid on every sweep, and a nudge before the sweep began is the
  # take's: neither is the arm's pull turning it over.
  from pluggybot.evaluation import probe as ep
  n = 1000
  phases = {"take": (0, 100), "up0": (100, 600), "down0": (600, n)}
  hinge, pin = np.zeros(n), np.zeros(n)
  pin[50:] = 0.3                                       # the take nudged it
  pin[150:300] = 0.3 - np.linspace(0.0, 0.6, 150)      # held: turned 0.6 rad back
  pin[300:] = -0.3
  hinge[400:] = ch.CAUGHT_OFF + np.linspace(1e-3, 1.0, n - 400)  # the catch let go...
  pin[400:600] = -0.3 - np.linspace(0.0, 1.0, 200)     # ...and it turns with the lid
  pin[600:] = -1.3
  assert ep.turned(pin, hinge, phases) == pytest.approx(math.degrees(0.6), abs=0.5)
  assert ep.turned(pin, np.zeros(n), phases) == pytest.approx(math.degrees(1.6), abs=0.5), \
    "a catch that never let go: the whole sweep up"
  assert ep.turned(pin, hinge, {"take": (0, 100)}) is None, "no sweep began"


def test_a_record_whose_handle_turned_over_is_flagged_and_kept(monkeypatch):
  # The flight's turn is read off OUR truth along the record (the lid's
  # angle, then the handle's), each replay's off its own rollout's joints;
  # past `TWISTED_DEG` the row names it, and its gaps stay in the row.
  from pluggybot.evaluation import probe as ep
  from pluggybot.imagination.rollout import Readings
  script = probe_chest()
  n = 600
  phases = {"take": (0, 100), "up0": (100, 400), "down0": (400, n)}

  def held_turn(rad):
    out = np.zeros(n)
    out[150:250] = np.linspace(0.0, rad, 100)
    out[250:] = rad
    return out
  turns = {"world": 0.1, "reference": 0.7}
  monkeypatch.setattr(ep.im, "truth_world", lambda lid, setting, pin: "world")
  monkeypatch.setattr(ep.im, "reference_world", lambda lid, setting, pin: "reference")
  monkeypatch.setattr(ep, "rollout", lambda world, record: Readings(
    t=np.zeros(n), sensed=np.zeros((n, 4)),
    joints={"pin": held_turn(turns[world]), "hinge": np.zeros(n)}))
  monkeypatch.setattr(ep.im, "compare", lambda truth, other, rows: [])
  out = SimpleNamespace(phases=phases, record=SimpleNamespace(
    n=n, dt=0.002, sensed=np.zeros((n, 4)), start=SimpleNamespace(pose=(24.0, -3.0, 0.0))))
  rows = np.column_stack([np.zeros(n), held_turn(0.55), np.ones(n), np.ones(n)])
  got = script.replayed(7, out, rows, (24.0, -3.0, 0.0), replay=True)
  assert got["turnedDeg"] == {"flight": pytest.approx(31.5, abs=0.1),
                              "world": pytest.approx(5.7, abs=0.1),
                              "reference": pytest.approx(40.1, abs=0.1)}
  assert got["twisted"] == ["flight", "reference"]
  assert set(got["replay"]) == {"world", "reference"}
  assert script.replayed(7, out, rows, (24.0, -3.0, 0.0), replay=False) == {
    "turnedDeg": {"flight": pytest.approx(31.5, abs=0.1)}, "twisted": ["flight"]}


class InlinePool:
  """`multiprocessing.Pool` in this process, saying how it was made."""
  made: list = []

  def __init__(self, processes, **kwargs):
    InlinePool.made.append(kwargs)

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def map(self, fn, jobs, chunksize=1):
    return [fn(j) for j in jobs]


def test_every_set_out_is_flown_in_a_process_of_its_own(monkeypatch):
  script = probe_chest()
  monkeypatch.setattr(script, "Pool", InlinePool)
  monkeypatch.setattr(InlinePool, "made", [])
  assert script.run_pool(abs, [-1, -2], 1) == [1, 2]
  assert InlinePool.made == [{"maxtasksperchild": 1}]


def test_a_set_out_that_raises_is_its_row_and_the_batch_keeps_the_rest(monkeypatch):
  # #479's review: `Pool.map` raises the first error a worker meets, and the
  # batch's rows went with it.
  from functools import partial
  script = probe_chest()
  monkeypatch.setattr(script, "Pool", InlinePool)

  def fly(args):
    if args[0] == 1:
      raise ValueError("zero-size array to reduction operation maximum")
    return {"k": args[0], "gotThrough": True}
  rows = script.run_pool(partial(script.guarded, fly), [(0,), (1,), (2,)], 1)
  assert [r["k"] for r in rows] == [0, 1, 2] and rows[2]["gotThrough"]
  assert rows[1]["error"].startswith("ValueError") and "gotThrough" not in rows[1]


def test_a_record_got_through_only_where_its_sweeps_were_flown(monkeypatch):
  # #479's review: a sweep that failed once the lid had opened read as got
  # through, and a take that failed -- no sweep at all -- crashed the
  # grading. The probe's own word that its sweeps were flown is asked first.
  script = probe_chest()
  n = 600
  monkeypatch.setattr(script.pr, "lid_from_arm", lambda out, spec: np.zeros(n))
  lid = np.r_[np.zeros(300), np.full(300, 1.1)]                # opened, its catch let go
  rows = np.column_stack([lid, np.zeros(n), np.ones(n), np.ones(n)])
  rec = SimpleNamespace(n=n, dt=0.002, depth=(), detections=())
  sweeps = {"take": (0, 100), "up0": (100, 350), "down0": (350, n)}

  def graded(phases, swept, opened=True):
    return script.graded(SimpleNamespace(record=rec, phases=phases, log={"swept": swept}),
                         rows, opened, None)
  assert graded(sweeps, True)["gotThrough"]
  assert not graded(sweeps, False)["gotThrough"]
  failed = graded({"take": (0, n)}, False, opened=False)
  assert (failed["gotThrough"], failed["grippedPct"], failed["lidFromArmDeg"]) == (False, None, None)
