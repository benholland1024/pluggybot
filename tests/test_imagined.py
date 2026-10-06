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
