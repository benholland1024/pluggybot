"""#775's measurement branch (never merged): the two switches the step's
cost at rest was measured on are off unless asked, and each is what it says.
"""
import mujoco
import numpy as np
import pytest

from pluggybot import sleeping
from pluggybot.activity import dishware as dw


def _robot_trees(m):
  return {int(m.body_treeid[m.body(n).id]) for n in ("pluggybot", "r2_pluggybot")}


def test_sleeping_is_off_unless_asked_and_then_no_robot_sleeps(monkeypatch):
  from pluggybot.humanoid import world as hw
  monkeypatch.delenv(sleeping.ENV, raising=False)
  m = hw.home_spec(second_at=(2.5, 0.5)).compile()
  assert not m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  monkeypatch.setenv(sleeping.ENV, "1")
  m = hw.home_spec(second_at=(2.5, 0.5)).compile()
  assert m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  assert m.opt.sleep_tolerance == sleeping.DEFAULT_TOLERANCE
  never = int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER)
  assert {int(m.tree_sleep_policy[t]) for t in _robot_trees(m)} == {never}
  plate = int(m.body_treeid[m.body("plate").id])
  assert m.tree_sleep_policy[plate] == int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_ALLOWED)


def test_the_counter_world_sleeps_too_where_asked(monkeypatch):
  from pluggybot.evaluation import counter as ct
  monkeypatch.setenv(sleeping.ENV, "1")
  m = ct.model_of(ct.drawer_set_out(0))
  assert m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  root = int(m.body_treeid[m.body("pluggybot").id])
  assert m.tree_sleep_policy[root] == int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER)


_PUSH = """<mujoco><option timestep="0.002" integrator="implicitfast"><flag sleep="enable"/></option>
<worldbody><geom type="plane" size="5 5 .1"/>
<body name="box" pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="0.3"/></body>
<body name="pusher" pos="-.3 0 .05" sleep="never"><joint name="slide" type="slide" axis="1 0 0"/>
<geom type="box" size=".03 .03 .03" mass="1"/></body></worldbody>
<actuator><position joint="slide" kp="400" kv="40"/></actuator></mujoco>"""


def test_a_sleeping_body_wakes_the_step_a_never_sleeping_one_touches_it():
  """The rule the measurement leans on (MuJoCo's Simulation chapter,
  "Waking"): a contact with an awake tree wakes a sleeping island in the
  position stage of the same step, and it moves as if it had never slept."""
  m = mujoco.MjModel.from_xml_string(_PUSH)
  d = mujoco.MjData(m)
  box = int(m.body_treeid[m.body("box").id])
  for _ in range(200):
    mujoco.mj_step(m, d)
  assert d.tree_asleep[box] >= 0, "a box at rest on the floor sleeps"
  x0 = float(d.xpos[m.body("box").id][0])
  d.ctrl[0] = 0.4
  touched = None
  for i in range(1000):
    mujoco.mj_step(m, d)
    pairs = {tuple(sorted((int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))))
             for c in d.contact[:d.ncon]}
    if (m.body("box").id, m.body("pusher").id) in pairs:
      touched = i if touched is None else touched
      assert d.tree_asleep[box] < 0, f"touched at step {i}, asleep"
  assert touched is not None
  assert float(d.xpos[m.body("box").id][0]) > x0 + 0.05, "pushed, it moved"


def test_the_raised_wall_keeps_the_served_plates_mass_and_inertia_to_the_bit():
  def plate(lift):
    spec = mujoco.MjSpec()
    dw.attach_plate(spec, (0.0, 0.0), 0.3, 0.0, 0.35, 0.7, tags=False, lift=lift)
    return spec.compile()
  a, b = plate(0.0), plate(0.0015)
  ia, ib = a.body("plate").id, b.body("plate").id
  for f in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
    assert np.array_equal(getattr(a, f)[ia], getattr(b, f)[ib]), f
  wall = b.geom("plate_wall0").id
  assert b.geom_pos[wall][2] - b.geom_size[wall][2] == pytest.approx(0.0015)
  assert b.geom_pos[wall][2] + b.geom_size[wall][2] == pytest.approx(dw.PLATE_H)
  assert dw.PLATE_WALL_LIFT == 0.0, "off unless PLUGGY_PLATE_WALL_LIFT"
