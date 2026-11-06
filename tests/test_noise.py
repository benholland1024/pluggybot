"""Each set-out flies noise of its own (#560, Ben's answer of 2026-11-06;
Evaluation.md §5): whose noise a body draws is ONE key, every stream of its
parts seeded off it (`humanoid.senses.Noise`); the attempt kit keys a
set-out by its job and number (`attempts.noise_of`), so a check's set-outs
no longer replay one realization of the robot's noise; a held-out set-out's
is sealed with its number; and the served robots keep their names'. Each
rule pinned without a flight (docs/Testing.md).
"""

import sys
from pathlib import Path
import zlib

import mujoco
import numpy as np
import pytest

from pluggybot.body import body_for
from pluggybot.evaluation import attempts as at
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import senses
from pluggybot.perception.imu import Imu, seed_for
from pluggybot.robot import SECOND

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _floor():
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def _state(rng) -> dict:
  return rng.bit_generator.state


def _streams(body, watch: bool = True) -> dict:
  """Every stream the body draws, where each has got to: the joints' shared
  block and each joint's own, each gripper's, the IMU's (and the offsets it
  drew), the LIDAR's and the near field's (each with its peers'), `measure`'s
  depth camera -- and, `watch`, a watcher's joints and depth, a frame read."""
  m = body.mission
  out = {"joints": _state(m.stand.joints.rng),
         **{f"joint {m.stand.joints.names[i]}": _state(r) for i, r in m.stand.joints.own},
         **{f"gripper {s}": _state(g.sense.rng) for s, g in m.reach.grippers.items()},
         "imu": (_state(m.stand.trunk.imu.rng), m.stand.trunk.imu.gyro_bias.tolist()),
         "lidar": (_state(m.lidar.rng), _state(m.lidar.peer_rng)),
         "near field": (_state(m.depth.rng), _state(m.depth.peer_rng))}
  next(body.depth_cloud(32, 24, 0.2))  # its camera built; its frame not yet taken
  out["measure"] = _state(body._depth[1].rng)
  if watch:
    body.depth_watched(32, 24)
    out.update(watched=_state(body._depth_watch[1].rng), watcher=_state(body._watch[0].rng))
  return out


def test_a_body_draws_every_stream_of_its_parts_off_its_one_key():
  """Two keys draw apart on every stream and one key twice the same. A
  stream left off the key would draw one realization under every key, as
  every set-out of a check once did (SimNotes, "The cube and the robot's
  noise")."""
  from pluggybot.humanoid.policy import StandPolicy
  from pluggybot.humanoid.walk import SEEING_NPZ
  a, b, again = (body_for(*_floor(), noise=k) for k in ("lid:0", "lid:1", "lid:0"))
  drew, other, twice = _streams(a), _streams(b), _streams(again)
  for name, got in drew.items():
    assert got != other[name], f"the {name}'s noise is one under two keys"
    assert got == twice[name], f"the {name}'s noise is not its key's"
  # ...and the camera of the walker that sees the ground (not served)
  seeing = StandPolicy(SEEING_NPZ)
  a, b, again = (_state(body_for(*_floor(), walk=seeing, noise=k).mission.head.cam.rng)
                 for k in ("lid:0", "lid:1", "lid:0"))
  assert a != b and a == again, "the ground's camera"


def test_the_robots_keep_their_names_noise():
  """A body no instrument keys draws every stream as the served pair always
  has, bit for bit: its parts off its name, its LIDAR off its root, its
  cameras off the seeds they were served with -- the second robot's off its
  prefix (`senses.Noise`, `SERVED_SEEDS`)."""
  model, data = _floor()
  crc = lambda s: zlib.crc32(s.encode())  # noqa: E731
  fresh = lambda seed: _state(np.random.default_rng(seed))  # noqa: E731
  imu = Imu("humanoid::pelvis")
  grips = {s: senses.GripperSense(model, s, key="humanoid:") for s in hm.SIDES}
  for g in grips.values():
    g.read(data)
  lidar = seed_for("lidar:pluggybot")
  assert _streams(body_for(*_floor()), watch=False) == {
    "joints": fresh(crc("joints:humanoid:")),
    "joint waist_yaw": fresh(crc("joints:humanoid::waist_yaw")),
    **{f"gripper {s}": _state(g.rng) for s, g in grips.items()},
    "imu": (_state(imu.rng), imu.gyro_bias.tolist()),
    "lidar": (fresh(lidar), fresh(lidar + 1)), "near field": (fresh(0), fresh(1)),
    "measure": fresh(466)}
  first, second = senses.Noise(), senses.Noise(SECOND)
  assert (first.watcher, first.seed("watched"), first.seed("ground")) == ("watch:", 530, 0)
  assert (second.name, second.watcher) == ("humanoid:r2_", "watch:r2_")
  assert [second.seed(s) for s in ("lidar", "near", "ground", "measure", "watched")] == [
    seed_for("lidar:r2_pluggybot"), 0, 0, 466, 530]
  assert senses.Noise(key=senses.ROBOT).name == "humanoid:"


def test_each_set_out_draws_noise_of_its_own_and_the_same_again(monkeypatch):
  """The attempt kit hands each set-out's body its job's and number's key
  (`attempts.noise_of`) -- every grader's, every set-out apart, one set-out
  the same twice -- or the override's, a draw flown again; and a reading
  says which its flights drew (`noise_said`)."""
  import pluggybot.body as body_mod
  handed = []

  class Built(Exception):
    pass

  def body_for(model, data, **kw):
    handed.append(kw.get("noise"))
    raise Built

  monkeypatch.setattr(body_mod, "body_for", body_for)

  def key(task, build=at.set_out):
    with pytest.raises(Built):
      build(task)
    return handed[-1]
  for grader in at.GRADERS:
    first, second, again = (key(at.Task(chest=k, grader=grader)) for k in (0, 1, 0))
    assert first != second and first == again, grader
  assert len({key(at.Task(chest=0, grader=g)) for g in at.GRADERS}) == len(at.GRADERS)
  assert key(at.Task(chest=1, grader="place"), at.builder("place:0")) == "place:0"
  assert key(at.Task(chest=1), at.builder(senses.ROBOT)) == senses.ROBOT

  def row(k, noise):
    return {"task": at.Task(chest=k).as_dict(), "run": {"noise": noise}}
  assert at.noise_said([row(0, "lid:0"), row(1, "lid:1"),
                        {**row(2, None), "error": "it raised"}]) == "each set-out drew noise of its own"
  assert at.noise_said([row(0, "lid:0"), row(1, None)]).startswith("⚠")
  assert at.noise_said([row(0, "robot"), row(1, "robot")]) == "every flight drew robot's noise"


def test_a_held_out_set_outs_noise_is_sealed_with_it(monkeypatch):
  """A held-out set-out's noise is its own number's, which the frozen build's
  seed drew (`held_out`): never a public set-out's, nor another seed's set-
  outs' -- and an instrument flying held-out set-outs refuses an override,
  which would fly every one of them on a draw somebody chose."""
  import attempts as attempts_script
  import hands_check
  a, b = (at.held_out(at.seed_of(c), 8) for c in ("0f98e24", "1ea5b16"))

  def keys(ks):
    return {at.noise_of(at.Task(chest=k)) for k in ks}
  assert len(keys(a)) == 8
  assert not keys(a) & keys(at.PUBLIC) and not keys(a) & keys(b)
  for script in (hands_check, attempts_script):
    monkeypatch.setattr(script, "run_pool", lambda *a, **kw: [])
    with pytest.raises(SystemExit):
      script.main(["--held-out", "2", "--seed", "1", "--noise", senses.ROBOT])
