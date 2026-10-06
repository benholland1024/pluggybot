"""The best-expressible reference (issue #466): the chest written in the scene
language by code that knows the truth, against the world's own chest.

The language says a lid's mass as the world has it, so what is left between
the two is what it cannot say -- the magnet's falloff, the contact, the
armature -- and with no catch the two behave alike along the oracle's probe.
`scripts/imagination_gap.py` measures the gap over drawn chests.
"""

from dataclasses import replace
import math

import mujoco
import pytest

from pluggybot.activity import chest as ch
from pluggybot.evaluation import imagined as im
from pluggybot.imagination.compile import PREFIX
from pluggybot.imagination.record import Record
from pluggybot.imagination.rollout import rollout
from pluggybot.imagination.scene import parse


def test_the_language_can_say_every_drawn_chest_wherever_it_stands():
  for k in range(16):
    for yaw in (0.0, 0.7, -2.0):
      s = parse(ch.reference_document(ch.draw(k), (1.0, -0.5), yaw, pin=-0.6))
      assert {j.id for j in s.joints} == {"hinge", "pin"}
      assert s.catches[0].release == ch.draw(k).catch_n


@pytest.mark.parametrize("yaw", [0.0, 0.9])
def test_the_reference_carries_the_lids_mass_as_the_world_has_it(yaw):
  # Mass shows on a hinge only as its moments (#469): the reference has the
  # world's, the handle hanging as it hangs; what it cannot say is its armature.
  lid, pin = ch.draw(5), -0.57
  setting = im.Setting(chest_x=0.6, chest_y=0.1, chest_yaw=yaw)
  truth = im.truth_world(lid, setting).model
  td = mujoco.MjData(truth)
  td.qpos[truth.jnt_qposadr[truth.joint("chest_pin").id]] = pin
  mujoco.mj_forward(truth, td)
  ref = im.reference_world(lid, setting, pin).model
  rd = mujoco.MjData(ref)
  mujoco.mj_forward(ref, rd)
  hinge = truth.joint("chest_hinge").id
  at, axis = td.xanchor[hinge], td.xaxis[hinge]
  rj = ref.joint(f"{PREFIX}hinge").id
  # (a document is written to the nanometre)
  assert rd.xanchor[rj] == pytest.approx(at, abs=1e-8)
  assert rd.xaxis[rj] == pytest.approx(axis, abs=1e-8)
  mine = im.mass_moments(ref, rd, ref.body(f"{PREFIX}lid").id, at, axis)
  theirs = im.mass_moments(truth, td, truth.body("chest_lid_body").id, at, axis)
  assert mine == pytest.approx(theirs, rel=1e-7)
  assert ref.dof_armature[ref.jnt_dofadr[rj]] == 0.0 < truth.dof_armature[truth.jnt_dofadr[hinge]]


def test_where_the_language_can_say_it_the_reference_behaves_as_the_world():
  # The oracle's probe, taken hold and two seconds into the first sweep up,
  # on a lid with no catch: the readings a hundredth of a newton apart at
  # the tool, below the 0.04-0.06 N the torques read (#469). A part out of
  # place, a mass or a joint wrong, and they part.
  lid = replace(ch.draw(1), catch_n=0.0)
  setting, pin = im.place(lid)
  record, phases = im.oracle_record(lid, setting, rates=(0.15,))
  cut = phases["up0"].start + 1000
  short = Record(start=record.start, dt=record.dt, commands=record.commands[:cut])
  truth = rollout(im.truth_world(lid, setting), short)
  ref = rollout(im.reference_world(lid, setting, pin), short)
  gap = {g.phase: g for g in im.compare(truth, ref, {"take": phases["take"],
                                                     "up": slice(phases["up0"].start, cut)})}
  assert math.degrees(truth.joints["hinge"][-1]) > 5.0          # the probe lifted it
  assert gap["take"].force_max < 0.01 and gap["up"].force_rms < 0.03
  assert gap["up"].lid_max < 0.2
  wrong = replace(lid, lid_kg=lid.lid_kg * 1.1)
  off = rollout(im.reference_world(wrong, setting, pin), short)
  far = im.compare(truth, off, {"up": slice(phases["up0"].start, cut)})[0]
  assert far.force_rms > 3 * gap["up"].force_rms


def test_the_oracle_holds_the_knob_where_it_hangs_and_stays_in_reach():
  lid = ch.draw(4)
  setting, pin = im.place(lid)
  world = im.truth_world(lid, setting)
  from pluggybot.imagination.rollout import settled
  d = settled(world, im.hold_record())
  knob = d.geom_xpos[world.model.geom("chest_knob").id]
  root = d.xpos[world.model.body("pluggybot").id]
  assert knob[0] - root[0] == pytest.approx(im.KNOB_AHEAD_M, abs=0.002)
  assert d.qpos[world.joints["pin"][0]] == pytest.approx(pin, abs=1e-9)
  record, phases = im.oracle_record(lid, setting)
  assert list(phases) == ["take", "up0", "down0", "up1", "down1"]


def test_the_references_parts_share_no_list():
  # Turning the lid alone turned all eight parts that shared its `euler`
  # list, each about its own middle, and tilting the pin's axis the hinge's
  # too (the review of #473).
  doc = ch.reference_document(ch.draw(0), (1.0, -0.5), 0.7, pin=-0.6)
  lists = [v for part in doc["parts"] + doc["joints"] for v in part.values()
           if isinstance(v, list)]
  assert len({id(v) for v in lists}) == len(lists)


def test_a_probe_sweeps_no_faster_than_the_chests_are_drawn_to_shut_at():
  # The fastest sweep down is what `draw`'s shut margin is held at: faster,
  # and a lid may push its handle back at the claw.
  assert max(im.RATES) == ch.SWEPT_RATE and im.TOP == ch.SWEPT_TO
  with pytest.raises(ValueError, match="drawn to shut"):
    im.oracle_record(ch.draw(0), im.Setting(chest_x=0.5), rates=(0.15, 0.6))


def test_a_phase_too_short_to_read_a_force_reads_none():
  import numpy as np
  from pluggybot.imagination.rollout import Readings
  n = 30
  quiet = Readings(t=np.arange(n) * 0.002, sensed=np.tile([0.5, 0.5, 2.2, 0.4], (n, 1)),
                   joints={"hinge": np.zeros(n)})
  pushed = Readings(t=quiet.t, sensed=quiet.sensed + [2.0, 2.0, 0, 0], joints=quiet.joints)
  [gap] = im.compare(quiet, pushed, {"short": slice(0, n)})
  assert gap.force_rms is None and gap.force_max is None and gap.lid_max == 0.0


def test_the_references_walls_are_where_the_chests_are():
  # The chest is written twice -- its MJCF and its document -- and the lid's
  # moments pin the moving half; the walls are pinned here, centre and size.
  import numpy as np
  setting = im.Setting(chest_x=0.6, chest_y=-0.2, chest_yaw=0.4)
  truth = im.truth_world(ch.Lid(), setting).model
  td = mujoco.MjData(truth)
  mujoco.mj_forward(truth, td)
  ref = im.reference_world(ch.Lid(), setting, 0.0).model
  rd = mujoco.MjData(ref)
  mujoco.mj_forward(ref, rd)
  for wall, part in (("floor", "floor"), ("back", "back"), ("front", "front"),
                     ("sider", "side_r"), ("sidel", "side_l")):
    t, r = truth.geom(f"chest_{wall}").id, ref.geom(f"{PREFIX}{part}").id
    assert np.allclose(rd.geom_xpos[r], td.geom_xpos[t], atol=1e-9), wall
    assert np.allclose(ref.geom_size[r], truth.geom_size[t], atol=1e-9), wall
