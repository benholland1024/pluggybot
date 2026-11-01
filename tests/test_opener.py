"""The reference opener's sight and path (`evaluation/opener.py`; SimNotes,
"The humanoid opens the chest"): the box guessed off the D435's cloud -- the
hinge, the drop handle's pin -- and the held knob's path as the lid turns.

The cloud is the probe's measuring view of the drawn chest (#466 stage 2, on
the quadruped until #559): a bare body carries the D435 at exactly the pose
the quadruped's head had there, and nothing of the robot was in that frame,
so these are the clouds and the guesses the probe's tests were written on.
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot.activity import chest as ch
from pluggybot.evaluation import opener as op
from pluggybot.perception import box
from pluggybot.perception.depth import DepthCamera
from pluggybot.robot import FIRST

FLOOR = ('<mujoco><compiler angle="radian" autolimits="true"/>'
         '<option timestep="0.002" integrator="implicitfast"/>'
         '<worldbody><geom name="floor" type="plane" size="20 20 0.1"/></worldbody></mujoco>')
#: A chest turned off the map's axes, so nothing lines up by accident.
CHEST = (1.2, 0.3, 0.35)
#: Its drop handle's turn on its pin at rest, rad (measured in the house).
REST_SWING = -0.653
#: The D435 standing 0.55 m off the knob, facing along the chest's axis: its
#: body's pose in the map and the camera's on it (the quadruped's, at
#: `quadruped-final`).
HEAD_POS, HEAD_QUAT = (0.6271230784173444, 0.09088359959159603, 0.3244), (
  0.9847265389049334, 0.0, 0.0, 0.17410813759359595)
EYE_POS, EYE_QUAT = (0.23325832229057822, 0.0, -0.006500013784713836), (
  0.6123726521413662, 0.3535530156982815, -0.3535530156982815, -0.6123726521413663)


@pytest.fixture(scope="module")
def scene():
  """The chest on a floor, the truth of where its joints are, and three of
  the D435's frames of it at the probe's resolution, in the map."""
  spec = mujoco.MjSpec.from_string(FLOOR)
  head = spec.worldbody.add_body(name=FIRST.root, pos=HEAD_POS, quat=HEAD_QUAT)
  head.add_camera(name="depth_eye", pos=EYE_POS, quat=EYE_QUAT, fovy=58.0)
  ch.attach_chest(spec, ch.Lid(), CHEST[:2], CHEST[2], tags=False)
  m = spec.compile()
  d = mujoco.MjData(m)
  # the handle as it rests: its centre of mass under its pin, swung 37.4 deg
  d.qpos[m.jnt_qposadr[m.joint("chest_pin").id]] = REST_SWING
  mujoco.mj_forward(m, d)
  cam = DepthCamera(m, FIRST, width=box.DEPTH_W, height=box.DEPTH_H,
                    min_z=box.DEPTH_MIN_Z, mount="body", seed=466)
  root = m.body(FIRST.root).id
  rot = d.xmat[root].reshape(3, 3)
  cloud = np.concatenate([cam.frame(d).points @ rot.T + d.xpos[root] for _ in range(3)])
  return SimpleNamespace(cloud=cloud, knob=d.geom_xpos[m.geom("chest_knob").id].copy(),
                         pin=d.xanchor[m.joint("chest_pin").id].copy(),
                         hinge=d.xanchor[m.joint("chest_hinge").id].copy())


def _u() -> np.ndarray:
  return np.array([math.cos(CHEST[2]), math.sin(CHEST[2]), 0.0])


def test_the_guess_finds_the_hinge_and_the_pin_off_depth(scene):
  # The hinge is the top's back edge at the top's height -- a board over the
  # true axis, the foresight's stated cost -- and the pin the tip of the
  # bracket's top, half the bracket over the pin.
  g = op.guess(scene.cloud, scene.knob, CHEST[2] + 0.2)
  assert isinstance(g, op.Guess), g
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


def test_the_bracket_tip_is_read_with_its_face_toward_the_camera(scene, monkeypatch):
  # Its front face stands its top at the tip: read off the mean of a window
  # (as a top's far edge is), the tip came 4 mm past it.
  monkeypatch.setattr(box, "faced_edge", lambda along, far=True: box.edge(along, far=far))
  g = op.guess(scene.cloud, scene.knob, CHEST[2] + 0.2)
  assert float(np.subtract(g.pin, scene.pin) @ _u()) < -0.003


def test_a_cloud_of_no_box_is_said_so():
  floor = np.column_stack([np.random.default_rng(0).uniform(0, 1, (500, 2)), np.zeros(500)])
  assert op.guess(floor, (0.5, 0.5, 0.1), 0.0) == "no top seen"


def test_a_held_knob_moves_as_its_pin_does_never_as_a_point_on_the_lid():
  # The knob hangs on a drop handle; held, the handle keeps its turn and the
  # knob rides the pin's arc. Planned as a point fixed to the lid -- the
  # knob's own arc about the hinge -- it is 62 mm off at the top.
  hinge = ch.HINGE
  pin = ch.HINGE + ch.PIN
  knob = pin + ch.KNOB_OFF
  axis = (0.0, 1.0, 0.0)
  path = op.pin_path(knob, pin, hinge, axis)
  assert np.allclose(path(0.0), knob)
  for s in (0.3, 0.8, op.TOP):
    turned_pin = hinge + op.rot_about(axis, s) @ (pin - hinge)
    assert np.allclose(path(s) - turned_pin, knob - pin), "the knob does not turn"
  fixed = hinge + op.rot_about(axis, op.TOP) @ (knob - hinge)
  assert np.linalg.norm(fixed - path(op.TOP)) > 0.06
