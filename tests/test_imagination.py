"""The robot's imagination (issue #466): the scene language, the world it
compiles into, the rollout, the worker and the fence.

The documents here are a cabinet with a lid and a drawer, not the chest:
nothing in the package is specific to a lid. Rollouts are short (a few
hundred rows) and the worker is one process for the module; the chest's own
gap is `tests/test_imagined.py`'s.
"""

import ast
import copy
import io
import math
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from pluggybot.imagination import compile as cp
from pluggybot.imagination import scene as sc
from pluggybot.imagination.record import COMMANDS, Record, Start, cloud
from pluggybot.imagination.rollout import rollout, settled
from pluggybot.imagination.worker import Imagination, pack, unpack
from pluggybot.legs import arm as am
from pluggybot.legs import rack as rk
from pluggybot.legs.model import CHOSEN, lie_qpos

SRC = Path(__file__).resolve().parents[1] / "src" / "pluggybot"

#: A cabinet half a metre in front of a robot at the map's origin: a lid
#: hinged at its back, a drawer that slides out toward the robot, its catch
#: at the drawer's knob.
CABINET = {
  "parts": [
    {"id": "carcass", "shape": "box", "size": [300, 400, 300], "pos": [750, 0, 150],
     "mass": 6.0},
    {"id": "lid", "shape": "slab", "size": [300, 400, 12], "pos": [750, 0, 306],
     "on": "carcass", "mass": 0.4},
    {"id": "drawer", "shape": "box", "size": [280, 360, 100], "pos": [745, 0, 80],
     "on": "carcass", "mass": 1.0},
    {"id": "knob", "shape": "cylinder", "size": [20, 30], "pos": [590, 0, 80],
     "euler": [0, 90, 0], "on": "drawer", "mass": 0.05}],
  "joints": [
    {"id": "lid_hinge", "type": "hinge", "part": "lid", "at": [900, 0, 300],
     "axis": [0, 1, 0], "range": [0, 100], "stiffness": 0.02, "slack": 140,
     "damping": 0.03, "friction": 0.05},
    {"id": "runner", "type": "slide", "part": "drawer", "at": [745, 0, 80],
     "axis": [-1, 0, 0], "range": [0, 200], "damping": 2.0, "friction": 0.5}],
  "catches": [{"joint": "runner", "at": [605, 0, 80], "release": 3.0}]}


def doc(**changes):
  out = copy.deepcopy(CABINET)
  out.update(changes)
  return out


def start(pose=(0.0, 0.0, 0.0)) -> Start:
  qs, qe = am.CARRY_Q
  return Start(pose=pose, attitude=(0.0, 0.0), legs=tuple(lie_qpos(CHOSEN)),
               arm=(qs, qs + qe), carrying="module_claw")


def hold(n: int = 200, pose=(0.0, 0.0, 0.0)) -> Record:
  st = start(pose)
  return Record(start=st, dt=0.002,
                commands=np.tile([st.arm[0], st.arm[1], 8.0, 1.5, 0.0, 0.0], (n, 1)))


@pytest.fixture(scope="module")
def cabinet():
  return cp.compile_scene(sc.parse(CABINET), carrying="module_claw")


@pytest.fixture(scope="module")
def worker():
  with Imagination(seed=7) as w:
    yield w


# ---- the language ---------------------------------------------------------------------

def test_a_document_is_read_in_the_authors_units_and_kept_in_the_sims():
  s = sc.parse(CABINET)
  lid, knob = s.part("lid"), s.part("knob")
  assert lid.size == pytest.approx((0.3, 0.4, 0.012))
  assert lid.pos == pytest.approx((0.75, 0.0, 0.306)) and lid.on == "carcass"
  assert knob.size == pytest.approx((0.02, 0.03)) and knob.euler[1] == pytest.approx(math.pi / 2)
  hinge, runner = s.joint("lid_hinge"), s.joint("runner")
  assert hinge.range == pytest.approx((0.0, math.radians(100)))
  assert hinge.slack == pytest.approx(math.radians(140)) and hinge.stiffness == 0.02
  assert runner.range == pytest.approx((0.0, 0.2)) and runner.axis == (-1.0, 0.0, 0.0)
  assert s.catches[0].release == 3.0 and s.catches[0].at == pytest.approx((0.605, 0.0, 0.08))


def test_the_round_trip_comes_back_to_the_same_document():
  s = sc.parse(CABINET)
  once = sc.dump(s)
  assert sc.dump(sc.parse(once)) == once
  again = sc.parse(once)
  for a, b in zip(s.parts, again.parts):
    assert a.id == b.id and a.shape == b.shape and a.on == b.on and a.mass == b.mass
    assert np.allclose(a.size + a.pos + a.euler, b.size + b.pos + b.euler, atol=1e-12)
  for a, b in zip(s.joints, again.joints):
    assert np.allclose(a.at + a.axis + a.range + (a.slack,),
                       b.at + b.axis + b.range + (b.slack,), atol=1e-12)
  assert again.catches == s.catches


@pytest.mark.parametrize("where, field", [
  ("doc", "solver"), ("part", "solref"), ("part", "friction"), ("joint", "armature"),
  ("joint", "solreflimit"), ("catch", "falloff")])
def test_an_unknown_field_is_refused_and_no_contact_parameter_has_one(where, field):
  # The imagination's contact is its own (`compile.CONTACT`): a document
  # that could set the world's would be a door for it.
  d = doc()
  target = {"doc": d, "part": d["parts"][1], "joint": d["joints"][0],
            "catch": d["catches"][0]}[where]
  target[field] = 1
  with pytest.raises(sc.Refused) as e:
    sc.parse(d)
  assert any(repr(field) in r for r in e.value.reasons)


def test_every_reason_comes_back_at_once():
  d = doc()
  d["parts"][1]["size"] = [300, 400, 120]          # a slab as thick as a box
  d["parts"][2]["on"] = "cupboard"                 # rides nothing there is
  d["parts"][3]["mass"] = 0                        # a part with no mass
  d["joints"][0]["range"] = [10, 100]              # leaves out the document's pose
  d["catches"][0]["release"] = -1
  with pytest.raises(sc.Refused) as e:
    sc.parse(d)
  text = " | ".join(e.value.reasons)
  for said in ("a slab is a board", "'cupboard', which is not a part", "more than 0 kg",
               "leaves out the pose", "release"):
    assert said in text
  assert len(e.value.reasons) >= 5


def test_what_rides_what_is_a_tree_and_a_part_has_one_joint():
  d = doc()
  d["parts"][0]["on"] = "knob"                     # round in a ring
  d["joints"].append({"id": "again", "type": "slide", "part": "drawer",
                      "at": [745, 0, 80], "axis": [0, 1, 0]})
  with pytest.raises(sc.Refused) as e:
    sc.parse(d)
  text = " | ".join(e.value.reasons)
  assert "comes back round" in text and "moved by two joints" in text


def test_a_catch_holds_its_joint_at_an_end_of_its_range_with_a_lever():
  d = doc(catches=[{"joint": "lid_hinge", "at": [900, 0, 300], "release": 2.0}])
  with pytest.raises(sc.Refused, match="no lever"):
    sc.parse(d)
  d = doc()
  d["joints"][1]["range"] = [-50, 200]
  with pytest.raises(sc.Refused, match="an end of its range"):
    sc.parse(d)


# ---- the world it compiles into --------------------------------------------------------

def test_a_scene_compiles_with_the_robots_own_body_and_where_the_document_says(cabinet):
  m = cabinet.model
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  # the robot's own: its body, its arm and the claw on its fork
  for name in ("pluggybot", "arm_upper", "module_claw"):
    m.body(name)
  s = sc.parse(CABINET)
  for p in s.parts:
    g = m.geom(f"{cp.PREFIX}{p.id}").id
    assert d.geom_xpos[g] == pytest.approx(p.pos, abs=1e-9), p.id
    assert m.body_mass[m.geom_bodyid[g]] == pytest.approx(p.mass)
  for j in s.joints:
    jid = m.joint(f"{cp.PREFIX}{j.id}").id
    assert d.xanchor[jid] == pytest.approx(j.at, abs=1e-9)
    assert d.xaxis[jid] == pytest.approx(j.axis, abs=1e-9)
    assert m.jnt_range[jid] == pytest.approx(j.range)
    dof = m.jnt_dofadr[jid]
    assert (m.dof_damping[dof], m.dof_frictionloss[dof]) == pytest.approx((j.damping, j.friction))
    assert m.jnt_stiffness[jid] == pytest.approx(j.stiffness)
  # nothing of a house: the floor, the robot, its tool and the scene
  assert {m.body(b).name for b in range(m.nbody) if m.body_parentid[b] == 0} == {
    "world", "pluggybot", "module_claw", *(f"{cp.PREFIX}{p.id}" for p in s.parts
                                          if p.on is None)}


def test_the_scene_touches_on_the_imaginations_own_contact(cabinet):
  # The world's furniture is MuJoCo's defaults; the imagination's is its
  # own, stiffer and slicker (#465's caution).
  m = cabinet.model
  for p in ("carcass", "lid", "knob"):
    g = m.geom(f"{cp.PREFIX}{p}").id
    assert m.geom_solref[g] == pytest.approx((0.01, 1.0))
    assert m.geom_friction[g][0] == pytest.approx(0.5)
  j = m.joint(f"{cp.PREFIX}lid_hinge").id
  assert m.jnt_solref[j] == pytest.approx((0.01, 1.0))
  default = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
  assert not np.allclose(default.geom_solref[0], m.geom_solref[m.geom(f"{cp.PREFIX}lid").id])


def test_a_moving_part_does_not_touch_what_it_is_hinged_to(cabinet):
  m = cabinet.model
  pairs = {frozenset((m.body(int(a)).name, m.body(int(b)).name))
           for a, b in zip(*np.divmod(m.exclude_signature, 1 << 16))}
  name = cp.PREFIX
  assert frozenset((f"{name}lid", f"{name}carcass")) in pairs
  assert frozenset((f"{name}drawer", f"{name}carcass")) in pairs
  assert frozenset((f"{name}knob", f"{name}carcass")) in pairs      # it rides the drawer
  assert frozenset((f"{name}lid", f"{name}drawer")) not in pairs


def test_a_compiled_catch_holds_below_its_release_and_lets_go_above_it(cabinet):
  catch = cabinet.hooks[0]
  assert catch.joint == "runner" and catch.sign == 1.0 and catch.lever == 1.0
  assert catch.force(0.0) == -3.0 and catch.force(cp.CATCH_REACH_M * 0.99) == -3.0
  assert catch.force(cp.CATCH_REACH_M * 1.01) == 0.0
  m = cabinet.model
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  drawer = m.body(f"{cp.PREFIX}drawer").id
  q = m.jnt_qposadr[m.joint(f"{cp.PREFIX}runner").id]
  for pull, lets_go in ((2.0, False), (4.0, True)):     # its friction is 0.5 N
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    for _ in range(300):
      d2.xfrc_applied[drawer, :3] = (-pull, 0.0, 0.0)
      mujoco.mj_step(m, d2)
      catch(m, d2)
    assert (d2.qpos[q] > cp.CATCH_REACH_M) == lets_go, pull


# ---- the record ------------------------------------------------------------------------

def test_a_record_carries_numbers_and_never_the_simulators_labels():
  # A depth frame carries the geom each pixel hit (`DepthFrame.peer_geoms`):
  # a cloud with a column for it is refused, as is anything not a number.
  pts = np.random.default_rng(0).normal(size=(5, 3))
  assert cloud(pts).shape == (5, 3)
  with pytest.raises(ValueError, match="k, 3"):
    cloud(np.column_stack([pts, np.arange(5)]))
  with pytest.raises(ValueError, match="numbers"):
    cloud(np.array([[mujoco.MjModel]] * 3, dtype=object))
  with pytest.raises(ValueError, match="plain data"):
    Record(start=start(), dt=0.002, commands=np.zeros((2, len(COMMANDS))),
           detections=({"id": 3, "geom": mujoco.MjData},))
  rec = Record(start=start(), dt=0.002, commands=np.ones((3, len(COMMANDS))),
               sensed=np.zeros((3, 4)), depth=(pts,),
               detections=({"id": 53, "t": [0.1, 0, 0.5]},))
  back = Record.from_wire(*rec.to_wire())
  assert np.array_equal(back.commands, rec.commands) and np.array_equal(back.depth[0], pts)
  assert back.start == rec.start and back.detections == rec.detections


# ---- the rollout -----------------------------------------------------------------------

def test_the_robot_starts_as_its_record_says_it_knew_itself(cabinet):
  m = cabinet.model
  rec = hold(pose=(0.1, -0.2, 0.3))
  d = settled(cabinet, rec)
  root = m.body("pluggybot").id
  rot = d.xmat[root].reshape(3, 3)
  assert d.xpos[root][:2] == pytest.approx((0.1, -0.2), abs=0.005)
  assert math.atan2(rot[1, 0], rot[0, 0]) == pytest.approx(0.3, abs=0.01)
  assert d.xpos[root][2] == pytest.approx(CHOSEN.belly_depth, abs=0.005)    # lying
  assert rk.tool_power(m, d, "module_claw")["powered"]                     # seated
  qs = d.qpos[m.jnt_qposadr[m.joint("arm_shoulder").id]]
  assert qs == pytest.approx(rec.start.arm[0], abs=0.02)


def test_a_rollout_is_the_same_twice_and_steps_at_the_worlds_rate(cabinet):
  rec = hold(150)
  a, b = rollout(cabinet, rec), rollout(cabinet, rec)
  assert np.array_equal(a.sensed, b.sensed) and a.t[0] == pytest.approx(0.002)
  assert set(a.joints) == {"lid_hinge", "runner"}
  slow = Record(start=rec.start, dt=0.004, commands=rec.commands)
  with pytest.raises(ValueError, match="a row a physics step"):
    rollout(cabinet, slow)


def test_noise_is_the_drivers_and_keyed_on_the_seed_never_the_worlds(cabinet):
  from pluggybot.perception.encoders import TORQUE_LSB, TORQUE_NOISE_NM
  rec = hold(400)
  exact = rollout(cabinet, rec)
  a, b = rollout(cabinet, rec, noise_seed=1), rollout(cabinet, rec, noise_seed=2)
  diff = a.sensed[:, :2] - exact.sensed[:, :2]
  assert 0.5 * TORQUE_NOISE_NM < diff.std() < 2 * TORQUE_NOISE_NM
  assert np.allclose(a.sensed[:, :2] / TORQUE_LSB, np.round(a.sensed[:, :2] / TORQUE_LSB))
  assert not np.array_equal(a.sensed, b.sensed)
  assert np.array_equal(a.sensed[:, 2:], exact.sensed[:, 2:])        # encoders untouched


# ---- the worker: the fence's process boundary -------------------------------------------

def test_a_message_carries_json_and_numbers_and_nothing_a_pickle_could(cabinet):
  with pytest.raises(TypeError):
    pack({"document": cabinet.model})
  with pytest.raises(TypeError):
    pack({}, {"world": np.array([cabinet.model], dtype=object)})
  with pytest.raises(TypeError):
    pack({}, {"data": mujoco.MjData(cabinet.model)})
  buf = io.BytesIO()
  np.savez(buf, __header__=np.frombuffer(b"{}", dtype=np.uint8),
           smuggled=np.array([{"model": 1}], dtype=object))
  with pytest.raises(ValueError):
    unpack(buf.getvalue())
  head, arrays = unpack(pack({"a": [1, 2.5]}, {"x": np.arange(3.0)}))
  assert head == {"a": [1, 2.5]} and np.array_equal(arrays["x"], np.arange(3.0))


def test_the_worker_rolls_out_what_the_process_would_bit_for_bit(worker, cabinet):
  rec = hold(150)
  here = rollout(cabinet, rec)
  there = worker.rollout(CABINET, rec)
  assert np.array_equal(here.sensed, there.sensed)
  assert all(np.array_equal(here.joints[k], there.joints[k]) for k in here.joints)
  with pytest.raises(sc.Refused, match="not a part"):
    worker.rollout(doc(parts=CABINET["parts"] + [{"id": "x", "shape": "box",
                                                    "size": [9, 9, 9], "pos": [0, 0, 0],
                                                    "on": "nowhere", "mass": 1}]), rec)
  assert np.array_equal(worker.rollout(CABINET, rec).sensed, here.sensed)   # still serving


def test_a_worker_is_deterministic_given_its_seed(worker):
  rec = hold(100)
  mine = worker.rollout(CABINET, rec, noisy=True)
  with Imagination(seed=7) as twin, Imagination(seed=8) as other:
    assert np.array_equal(twin.rollout(CABINET, rec, noisy=True).sensed, mine.sensed)
    assert not np.array_equal(other.rollout(CABINET, rec, noisy=True).sensed, mine.sensed)


# ---- the fence: the imports ---------------------------------------------------------------

#: What the robot's model may never be built from: the world's mechanisms
#: (`activity`), its challenges and its house, our grading of it
#: (`evaluation`), and the spike that chose the chest.
FENCED = ("pluggybot.activity", "pluggybot.challenge", "pluggybot.home",
          "pluggybot.legs.world", "pluggybot.evaluation", "mechanism_spike")


def test_the_imagination_imports_nothing_of_the_worlds_mechanisms():
  files = sorted((SRC / "imagination").rglob("*.py"))
  assert {f.name for f in files} >= {"scene.py", "compile.py", "record.py", "rollout.py",
                                     "worker.py"}
  for path in files:
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
      names = ([a.name for a in node.names] if isinstance(node, ast.Import)
               else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
      assert not any(n.startswith(f) for n in names for f in FENCED), (path, names)
    for word in ("activity.chest", "activity/chest", "mechanism_spike", "import_module",
                 "__import__"):
      assert word not in text, (path, word)


def test_nothing_the_imagination_loads_reads_the_worlds_mechanisms():
  # Transitively: a module imported by one the imagination imports.
  code = ("import sys\n"
          "import pluggybot.imagination.worker, pluggybot.imagination.compile\n"
          "from pluggybot.imagination import rollout\n"
          "from pluggybot.imagination.compile import robot_spec\n"
          "robot_spec('module_claw')\n"
          "print(' '.join(m for m in sys.modules if m.startswith('pluggybot')))\n")
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       check=True).stdout.split()
  assert "pluggybot.imagination.rollout" in out
  assert not [m for m in out if m.startswith(FENCED)], out


def test_the_languages_own_example_parses():
  import json
  doc = sc.__doc__
  start = doc.index('{"parts"')
  example = json.loads(doc[start:doc.index("]}\n", start) + 2])
  s = sc.parse(example)
  assert [p.id for p in s.parts] == ["base", "lid"] and s.catches[0].joint == "hinge"
