"""The carry instrument (`scripts/carry_spike.py`, #560's next task): what it
reports is walk2 walking with an arm posed holding a piece as our table
reference holds it -- so its grips must BE the reference's, its wrist must
hold the piece level, and its hand-over must leave the arm where it posed it
(the served hand-over opens the hand and brings it home, #669: walked so, the
instrument would report arms home)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import carry_spike as cs  # noqa: E402

from pluggybot.evaluation import hands_check as hc  # noqa: E402
from pluggybot.humanoid import model as hm  # noqa: E402


def _turn(f, j) -> str:
  return ", ".join(f"{v:g}" for v in (*f, *j))


def test_its_grips_are_the_table_references():
  # the plate's left hand and the cup's right, each turned as the reference
  # asks them -- a reference asking another grip leaves `PINCH` measured on
  # one that is no longer ours
  for piece, side, at in (("plate", "left", "reach(\"left\""), ("cup", "right", "reach(\"right\"")):
    p_side, _, _, f, j, _ = cs.PIECE[piece]
    assert p_side == side
    lines = [ln for ln in hc.TABLE.splitlines() if at in ln]
    assert any(_turn(f, j) in ln.replace(".0,", ",") for ln in lines), (piece, lines)
  # ...the plate's wall gripped 6.6 cm out of its middle at the reference's height
  assert "gy = py + 0.066 * (1 - i)" in hc.TABLE and "gz = 0.7734" in hc.TABLE
  assert cs.PIECE["plate"][2][1] == 0.066 and math.isclose(cs.TABLE_H + cs.PIECE["plate"][2][2], 0.7734)


def test_its_wrist_holds_each_piece_level():
  for piece in ("plate", "cup"):
    side = cs.HELD[piece][0][1]
    q = cs.pose_joints("tray", side, piece)
    tilt, _, _ = cs._tilt_at(side, q, piece)
    assert tilt < 0.5, (piece, tilt)
  # ...and a wrist left hanging does not: the solve is what levels it
  q = cs.pose_joints("tray", "left", "plate")
  assert cs._tilt_at("left", {**q, "wrist_pitch": 0.0, "wrist_roll": 0.0}, "plate")[0] > 20.0


def test_its_hand_over_leaves_the_arm_where_it_is_posed():
  model, data, body, info = cs.build("empty_out", 0, "test")
  try:
    s = cs.setup(body, model, data, info, None)
    assert s["ok"], s
    mi = body.mission
    q0 = mi.reach.q_cmd["left"].copy()
    course = cs.course_routine(body, "straight", None, mi.falls)
    command = next(course)
    for _ in range(int(round(0.6 / model.opt.timestep))):
      body.stepper.step(command)
      command = course.send(None)
    assert mi.legs == "walk" and mi.reach.phase["left"] == "joints"
    assert np.allclose(mi.reach.q_cmd["left"], q0)
    q, _, _ = mi.stand.joints.read(data)
    arm = [hm.JOINT_NAMES.index(f"left_{n}") for n in hm.ARM]
    assert np.abs(q[arm] - q0).max() < 0.15
    course.close()
  finally:
    body.close()
