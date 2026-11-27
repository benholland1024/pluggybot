"""MuJoCo's island sleeping (#779; `pluggybot.sleeping`): on in every world
the house's and the counter's builders make, never a robot, and its state
carried through a restore so that the world steps on bit for bit -- through
a tree falling asleep, and through one waking. The served world's restarts
are pinned in `tests/test_continuation.py`; here, on a toy world, every way
a step can cross a sleep or a wake, in microseconds a step."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import continuation, sleeping

#: Three boxes at rest (two touching: one island, a cycle of two), a
#: cupboard and its lever on springs (a tree of two bodies), a static shelf,
#: a mocap target, and a pusher that never sleeps, driven into the boxes.
TOY = """<mujoco>
  <option timestep="0.002" integrator="implicitfast"><flag sleep="enable"/></option>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="shelf" pos="2 0 0"><geom type="box" size=".2 .2 .05" pos="0 0 .05"/></body>
    <body name="target" mocap="true" pos="0 3 1">
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/></body>
    <body name="alpha" pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".3"/></body>
    <body name="beta" pos=".1 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".3"/></body>
    <body name="gamma" pos=".5 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".3"/></body>
    <body name="cupboard" pos="1 1 .3">
      <joint name="hinge" type="hinge" axis="0 0 1" stiffness="2" springref=".1" damping=".05"/>
      <geom type="box" size=".2 .02 .2" contype="0" conaffinity="0" mass="1"/>
      <body name="lever" pos=".15 .03 0">
        <joint name="pivot" type="hinge" axis="0 1 0" stiffness=".5" damping=".01" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 .05 0 0" size=".01" contype="0" conaffinity="0" mass=".05"/>
      </body>
    </body>
    <body name="pusher" pos="-.4 0 .05" sleep="never">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".03 .03 .03" mass="1"/>
    </body>
  </worldbody>
  <actuator><position name="push" joint="slide" kp="400" kv="40"/></actuator>
  <sensor><jointpos joint="hinge"/><jointpos joint="pivot"/>
    <framepos objtype="body" objname="gamma"/></sensor>
</mujoco>"""

#: The toy's day, flown: the three boxes asleep at step 76; alpha and beta
#: woken by the pusher's touch at 360, gamma by beta's at 518; the cupboard
#: asleep at 692. Each a step a save can fall on, either side of it.
STEPS = 900
EVENTS = (76, 360, 518, 692)


def _toy(text: str = TOY):
  m = mujoco.MjModel.from_xml_string(text)
  return m, mujoco.MjData(m)


def _ctrl(i: int) -> float:
  return 0.0 if i < 150 else min(0.9, (i - 150) * 0.002)


def _tree(m, name: str) -> int:
  return int(m.body_treeid[m.body(name).id])


def _seen(d) -> tuple:
  return (d.qpos.copy(), d.qvel.copy(), np.array(d.tree_asleep), d.sensordata.copy(),
          d.qacc_warmstart.copy(), d.xpos.copy())


def _fly(m, d, start: int, stop: int, saves=(), hook=None) -> tuple[list, dict]:
  """The toy's schedule from step `start` to `stop`: what each step left,
  and the world saved after each step of `saves`, as the keeper saves it
  (its last step kept)."""
  last = continuation.LastStep(SimpleNamespace(model=m, data=d))
  seen, saved = [], {}
  for i in range(start, stop):
    d.ctrl[0] = _ctrl(i)
    mujoco.mj_step(m, d)
    if hook is not None:
      hook(i, d)
    last.hook()
    seen.append(_seen(d))
    if i in saves:
      saved[i] = continuation.physics(m, d, last)
  return seen, saved


def _same(a: tuple, b: tuple) -> bool:
  return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_every_world_sleeps_at_mujocos_tolerance_and_no_robot_ever_does():
  """The served house with its pair and a counter's world, as their builders
  make them: the flag on at MuJoCo's own tolerance, every robot's tree
  `never` -- through a redraw's `mj_setConst` too -- and everything else
  the compiler's (the plate may sleep; a station's plug, on its balancer,
  never). Stepped bare, what rests is asleep in 0.2 s, the robots never."""
  from pluggybot.activity import dishware
  from pluggybot.evaluation import counter
  from pluggybot.humanoid.world import home_spec
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  pair = home_spec(second_at=world_config(HUMANOID_HOME)["start2"][:2]).compile()
  alone = counter.model_of(counter.drawer_set_out(0))
  never = int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER)
  for m, robots in ((pair, ("pluggybot", "r2_pluggybot")), (alone, ("pluggybot",))):
    assert sleeping.on(m)
    assert m.opt.sleep_tolerance == mujoco.MjSpec().option.sleep_tolerance
    assert [int(m.tree_sleep_policy[_tree(m, r)]) for r in robots] == [never] * len(robots)
  dishware.set_const(pair)
  assert int(pair.tree_sleep_policy[_tree(pair, "r2_pluggybot")]) == never
  allowed = int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_ALLOWED)
  assert int(pair.tree_sleep_policy[_tree(pair, "plate")]) == allowed
  assert int(pair.tree_sleep_policy[_tree(pair, "station_plug")]) != allowed
  d = mujoco.MjData(pair)
  robots = [_tree(pair, r) for r in ("pluggybot", "r2_pluggybot")]
  for _ in range(100):
    mujoco.mj_step(pair, d)
    assert all(d.tree_asleep[t] < 0 for t in robots)
  for name in ("block_0", "mass_known", "cube", "drawer_drawer", "plate", "fork"):
    assert d.tree_asleep[_tree(pair, name)] >= 0, name


def test_update_is_mujocos_own_mj_update_sleep():
  """`sleeping.update` rebuilds what MuJoCo derives from `tree_asleep`
  exactly as MuJoCo does (its `mj_updateSleep`, which the bindings do not
  expose): awake, asleep, a static body, a mocap body, a tree of two
  bodies, a never tree -- read at rest, with the boxes asleep, and with
  the cupboard asleep beside the woken boxes."""
  m, d = _toy()
  counts = ("ntree_awake", "nbody_awake", "nparent_awake", "nv_awake")
  for at in (0, 100, 800):
    while int(np.round(d.time / m.opt.timestep)) < at:
      d.ctrl[0] = _ctrl(int(np.round(d.time / m.opt.timestep)))
      mujoco.mj_step(m, d)
    want = {k: getattr(d, k) for k in counts}
    arrays = {k: np.array(getattr(d, k)) for k in ("tree_awake", "body_awake", "body_awake_ind",
                                                     "parent_awake_ind", "dof_awake_ind")}
    for k in arrays:
      getattr(d, k)[:] = 7
    for k in counts:
      setattr(d, k, 0)
    sleeping.update(m, d)
    assert {k: getattr(d, k) for k in counts} == want
    assert np.array_equal(d.tree_awake, arrays["tree_awake"])
    assert np.array_equal(d.body_awake, arrays["body_awake"])
    for k, n in (("body_awake_ind", "nbody_awake"), ("parent_awake_ind", "nparent_awake"),
                 ("dof_awake_ind", "nv_awake")):
      assert np.array_equal(getattr(d, k)[:want[n]], arrays[k][:want[n]]), k
  assert d.tree_asleep[_tree(m, "cupboard")] >= 0 and d.tree_asleep[_tree(m, "alpha")] < 0


def test_a_touch_wakes_a_sleeping_body_in_the_step_it_lands():
  """The rule sleeping leans on (MuJoCo's "Waking"): a contact with an
  awake tree wakes a sleeping island in the position stage of the step it
  lands in, and it moves."""
  m, d = _toy()
  alpha, pusher = m.body("alpha").id, m.body("pusher").id
  touched = None
  for i in range(EVENTS[1] + 50):
    d.ctrl[0] = _ctrl(i)
    mujoco.mj_step(m, d)
    pairs = {tuple(sorted((int(m.geom_bodyid[a]), int(m.geom_bodyid[b]))))
             for a, b in d.contact.geom[:d.ncon]}
    if (alpha, pusher) in pairs and touched is None:
      touched = i
      assert d.tree_asleep[_tree(m, "alpha")] < 0, f"touched at step {i}, asleep"
    if i == EVENTS[1] - 5:
      assert d.tree_asleep[_tree(m, "alpha")] >= 0, "the premise: asleep before the touch"
  assert touched == EVENTS[1]
  assert float(d.xpos[alpha][0]) > 0.05, "pushed, it moved"


@pytest.mark.parametrize("event", EVENTS)
def test_a_save_on_any_step_round_a_sleep_or_a_wake_steps_on_bit_for_bit(event):
  """A save the step before, the step of and the step after each sleep and
  wake of the toy's day, restored into a fresh world -- its last step
  stepped again (`replay`) -- steps on as the world flown straight through:
  every position, velocity, sleep, sensor and warm start, to the day's end.
  Put back with every island awake, as MuJoCo's own state calls put it, the
  first save after the boxes slept parts at once."""
  saves = (event - 1, event, event + 1)
  m, d = _toy()
  seen, saved = _fly(m, d, 0, STEPS, saves)
  for k in saves:
    index, arrays = saved[k]
    back = mujoco.MjData(m)
    continuation.put_physics(m, back, index, arrays)
    assert continuation.replay(m, back, index, arrays), k
    assert _same(_seen(back), seen[k]), k
    again, _ = _fly(m, back, k + 1, STEPS)
    for i, (a, b) in enumerate(zip(again, seen[k + 1:])):
      assert _same(a, b), (k, k + 1 + i)


def test_a_replay_lands_on_the_saves_sleep_whatever_its_wake_counted():
  """A tree woken by a toucher still counting down to sleep takes the
  toucher's count; stepped again from where the step began, the pose it
  woke at wakes it, counting from the start -- the same motion, another
  count, and it sleeps later than the world flown straight through. So a
  replay that lands where the save stood takes the save's sleep. Beta,
  moving, is set counting as a slow toucher's would be the step before it
  touches gamma."""
  m, d = _toy()
  beta, gamma = _tree(m, "beta"), _tree(m, "gamma")

  def counting(i, d):
    if i == EVENTS[2] - 1:
      d.tree_asleep[beta] = -5

  seen, saved = _fly(m, d, 0, STEPS, (EVENTS[2],), hook=counting)
  index, arrays = saved[EVENTS[2]]
  assert arrays["before/sleep"][gamma] >= 0 and -10 < arrays["sleep"][gamma] < 0, "the premise"
  back = mujoco.MjData(m)
  continuation.put_physics(m, back, index, arrays)
  assert continuation.replay(m, back, index, arrays)
  assert np.array_equal(back.tree_asleep, arrays["sleep"])
  again, _ = _fly(m, back, EVENTS[2] + 1, STEPS)
  assert all(_same(a, b) for a, b in zip(again, seen[EVENTS[2] + 1:]))


def test_the_sleep_state_is_put_back_by_name_and_a_cycle_this_world_lacks_wakes():
  """A world that lists its trees the other way round gets each tree's
  sleep back by its root body's name, the island's cycle remapped; one that
  lacks a tree of a sleeping cycle wakes the rest of it."""
  m, d = _toy()
  for i in range(100):
    mujoco.mj_step(m, d)
  names, asleep = sleeping.kept(m, d)
  assert asleep[names.index("alpha")] >= 0 and asleep[names.index("beta")] >= 0
  swapped = TOY.replace('name="alpha"', 'name="tmp"').replace('name="beta"', 'name="alpha"') \
               .replace('name="tmp"', 'name="beta"')
  m2, d2 = _toy(swapped)
  sleeping.put(m2, d2, names, asleep)
  order = sleeping.keys(m2)
  cycle = {order[t]: order[int(d2.tree_asleep[t])] for t in range(m2.ntree) if d2.tree_asleep[t] >= 0}
  assert cycle["alpha"] == "beta" and cycle["beta"] == "alpha" and cycle["gamma"] == "gamma"
  assert int(d2.ntree_awake) == m2.ntree - 3
  lacking = TOY.replace('name="beta"', 'name="delta"')
  m3, d3 = _toy(lacking)
  sleeping.put(m3, d3, names, asleep)
  assert d3.tree_asleep[_tree(m3, "alpha")] == sleeping.AWAKE
  assert d3.tree_asleep[_tree(m3, "gamma")] >= 0


def test_a_put_state_is_forwarded_with_every_tree_awake_its_sleep_put_back_after():
  """A sleeping tree's derived quantities are never computed again until it
  wakes, and one whose pose is put back as it was is not woken by MuJoCo:
  so a put state is forwarded with every tree awake, and only then put to
  sleep -- or a box redrawn where it lay (`storeroom.redraw` draws a prop's
  mass) kept its old mass in its inertia until something touched it."""
  m, d = _toy()
  for _ in range(100):
    mujoco.mj_step(m, d)
  index, arrays = continuation.physics(m, d)
  gamma = m.body("gamma").id
  assert d.tree_asleep[_tree(m, "gamma")] >= 0, "the premise: asleep where it lies"
  m.body_mass[gamma] = 0.5
  continuation.put_physics(m, d, index, arrays)
  assert d.tree_asleep[_tree(m, "gamma")] >= 0
  assert d.cinert[gamma][9] == 0.5
