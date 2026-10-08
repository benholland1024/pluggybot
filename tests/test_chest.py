"""The drop-handle chest (issue #466): demo 1's mechanism, as an activity.

Its draw is #469's set-outs with a catch, each kept only where its lid shuts
by the margin a drop handle needs; its flags come off its sensor with
hysteresis and a latch; its catch holds a pull below its own and lets go
above it. The physics is the chest alone on a floor, no robot.
"""

from dataclasses import replace
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.activity import chest as ch

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ('<mujoco><compiler angle="radian"/><option timestep="0.002"/>'
         '<worldbody><geom name="floor" type="plane" size="2 2 0.1"/></worldbody></mujoco>')


def chest_world(lid: ch.Lid, tags: bool = False):
  spec = mujoco.MjSpec.from_string(FLOOR)
  if tags:
    spec.modelfiledir = str(ROOT / "models")
  ch.attach_chest(spec, lid, (0.0, 0.0), tags=tags)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data, ch.Chest(model, data, lid)


# ---- the hidden parameters ---------------------------------------------------------

def test_a_draw_is_the_spikes_set_out_and_comes_back_for_its_seed():
  # #469 flew set-outs 0-7 of its own draw (`mechanism_spike.draw`, before
  # the chest was promoted): the chest's are those, every field, so the
  # spike's numbers describe the chests stage 2 flies. A catch is drawn
  # after them.
  spike = {
    0: ch.Lid(lid_kg=0.30670505560205075, lump_kg=0.009010072921954332,
              lump_at=0.8510954118888623, stiffness=0.0, springref=2.5,
              damping=0.007343943561794442, friction=0.04126889127145848,
              catch_n=3.8882470660994954),
    1: ch.Lid(lid_kg=0.43981800513069536, lump_kg=0.09272442493967145,
              lump_at=0.4481575599093507, stiffness=0.01792967507716778, springref=2.5,
              damping=0.017163757908979105, friction=0.02314397662942881,
              catch_n=3.715564190304792),
    7: ch.Lid(lid_kg=0.4035797700275241, lump_kg=0.05670873163635046,
              lump_at=0.4289378390320254, stiffness=0.010736554406217307, springref=2.5,
              damping=0.0280518718752657, friction=0.04365925094697502,
              catch_n=2.246951178186034)}
  for k, lid in spike.items():
    assert ch.draw(k) == lid
    assert ch.draw(k) == ch.draw(k)
  assert ch.draw(0) != ch.draw(1)
  with pytest.raises(ValueError, match="seed"):
    ch.draw(-1)


def test_every_lid_drawn_shuts_by_the_margin_a_drop_handle_needs():
  # A drop handle only pulls: a lid whose spring beat its weight near the
  # top pushed the handle back at the claw, which swung 9-14 deg (#469's
  # `open` corner) -- the premise, and every draw kept clear of it.
  th = np.linspace(0.0, ch.SWEPT_TO, 25)
  corner = ch.Lid(lid_kg=0.30, stiffness=0.15, springref=ch.SPRING_SLACK)
  assert ch.closing_torque(corner, th, ch.SWEPT_RATE).min() < ch.CLOSING_MARGIN_NM
  for k in range(40):
    lid = ch.draw(k)
    assert ch.closing_torque(lid, th, ch.SWEPT_RATE).min() >= ch.CLOSING_MARGIN_NM
    for name, (lo, hi) in ch.RANGES.items():
      assert lo <= getattr(lid, name) <= hi, (k, name)
    assert ch.CATCH_RANGE[0] <= lid.catch_n <= ch.CATCH_RANGE[1]
    assert (lid.stiffness == 0.0) == (k % 2 == 0)


def test_the_catch_pulls_shut_and_falls_with_the_gap_to_nothing():
  lid = ch.Lid(catch_n=2.0)
  cut = ch.CATCH_CUT / ch.PIN_R
  pulls = [ch.catch_torque(lid, a) for a in np.linspace(0.0, cut, 20)]
  assert pulls[0] == pytest.approx(-2.0 * (1 - 11 ** -2) * ch.PIN_R)
  assert all(a <= b for a, b in zip(pulls, pulls[1:]))
  assert pulls[-1] == pytest.approx(0.0, abs=1e-12)
  assert ch.catch_torque(lid, cut * 1.01) == 0.0
  assert ch.catch_torque(lid, -0.001) == pulls[0]
  assert ch.catch_torque(replace(lid, catch_n=0.0), 0.0) == 0.0


# ---- the flags ------------------------------------------------------------------------

def test_the_lid_opens_and_shuts_on_hysteresis_and_opened_latches():
  model, data, chest = chest_world(ch.Lid())
  assert chest.flags == {"lid": "shut", "opened": False, "caught": True}
  seen = []
  for deg in (0, 0.6, 1.2, 0.8, 0.3, 7, 12, 8, 4, 60, 30, 3, 0):
    chest.update(math.radians(deg))
    seen.append((deg, dict(chest.flags)))
  flags = dict(seen)
  # the catch's reach is 5 mm at the knob (1.06 deg); it holds again at half
  assert flags[0.6]["caught"] and not flags[1.2]["caught"] and not flags[0.8]["caught"]
  assert flags[0.3]["caught"]
  # open past 10 deg, shut again only under 5
  assert flags[7]["lid"] == "shut" and flags[12]["lid"] == "open"
  assert flags[8]["lid"] == "open" and flags[4]["lid"] == "shut"
  # past 57 deg it has been opened, for good
  assert not flags[12]["opened"] and flags[60]["opened"] and flags[0]["opened"]


def test_the_flags_come_off_the_sensor_and_the_magnets_pull_off_the_lid():
  lid = ch.draw(3)
  model, data, chest = chest_world(lid)
  hinge = model.jnt_qposadr[model.joint("chest_hinge").id]
  data.qpos[hinge] = 0.5                # moved, and nothing has read it yet
  chest.sense(model, data)
  assert chest.flags["lid"] == "shut"
  assert data.qfrc_applied[chest.dof] == 0.0          # out of the magnet's reach
  data.sensordata[chest.sensor_adr] = 0.5
  data.qpos[hinge] = 0.0
  chest.sense(model, data)
  assert chest.flags["lid"] == "open"
  # the pull is physics: the lid's angle as it is, never a sensor's
  assert data.qfrc_applied[chest.dof] == ch.catch_torque(lid, 0.0) < 0
  assert set(chest.flags) == {"lid", "opened", "caught"}
  assert all(isinstance(v, (bool, str)) for v in chest.flags.values())


def test_a_restart_keeps_the_latch_and_which_side_of_the_hysteresis():
  import json
  model, data, chest = chest_world(ch.Lid())
  for deg in (12, 60, 7, 0.6, 1.2, 0.8):
    chest.update(math.radians(deg))
  saved = json.loads(json.dumps(chest.kept_state()))
  fresh = ch.Chest(model, data, ch.Lid())
  fresh.restore_kept(saved)
  for c in (chest, fresh):
    c.update(math.radians(0.8))
  assert fresh.flags == chest.flags == {"lid": "shut", "opened": True, "caught": False}


def test_the_chest_rests_shut_and_its_catch_lets_go_only_past_its_pull():
  # Pulled up at its pin, ramped: the lid leaves the catch where the pull's
  # torque passes what holds it shut -- its weight, the catch and its
  # friction -- and not before.
  lid = replace(ch.draw(2), catch_n=3.0)
  model, data, chest = chest_world(lid)
  for _ in range(250):
    mujoco.mj_step(model, data)
    chest.sense(model, data)
  assert chest.flags == {"lid": "shut", "opened": False, "caught": True}
  body = model.body("chest_lid_body").id
  hinge = model.jnt_qposadr[model.joint("chest_hinge").id]
  pin = data.xpos[model.body("chest_handle").id].copy()
  # what holds it shut, N*m about the hinge: the lid and what hangs off its pin
  first = sum(float(model.body_mass[b]) * float(data.xipos[b][0] - data.xpos[body][0])
              for b in range(model.nbody) if _under(model, b, body))
  held = -ch.G * first + lid.catch_n * (1 - 11 ** -2) * ch.PIN_R + lid.friction
  lever = float(data.xpos[body][0] - pin[0])
  released_at = None
  force = 0.0
  while released_at is None and force < 2 * held / lever:
    force += 2.0 * model.opt.timestep           # 2 N/s
    data.xfrc_applied[body, :3] = (0.0, 0.0, force)
    data.xfrc_applied[body, 3:] = np.cross(pin - data.xipos[body], (0.0, 0.0, force))
    mujoco.mj_step(model, data)
    chest.sense(model, data)
    if not chest.flags["caught"]:
      released_at = force
  assert data.qpos[hinge] > 0
  assert released_at == pytest.approx(held / lever, rel=0.05)


def _under(model, b: int, root: int) -> bool:
  while b not in (0, root):
    b = int(model.body_parentid[b])
  return b == root


# ---- the tag ---------------------------------------------------------------------------

def test_two_chests_in_one_world_share_the_knobs_one_tag():
  spec = mujoco.MjSpec.from_string(FLOOR)
  spec.modelfiledir = str(ROOT / "models")
  ch.attach_chest(spec, ch.Lid(), (0.0, 0.0))
  ch.attach_chest(spec, ch.draw(1), (0.0, 0.8), name="chest2")
  model = spec.compile()
  assert model.ntex == 1 and model.body("chest2_lid_body") is not None


def test_the_knob_wears_its_tag_and_the_committed_png_is_the_generators():
  from PIL import Image
  from pluggybot.rack.tags import TAG_DIR, TAG_SIZES, BLOCK_TAG_SIZE, tag_image
  png = np.asarray(Image.open(ROOT / TAG_DIR / f"tag{ch.KNOB_TAG_ID}.png"))
  assert np.array_equal(png, tag_image(ch.KNOB_TAG_ID))
  assert TAG_SIZES[ch.KNOB_TAG_ID] == BLOCK_TAG_SIZE
  model, _, _ = chest_world(ch.Lid(), tags=True)
  knob = model.geom("chest_knob").id
  assert model.mat(model.geom_matid[knob]).name == f"tagmat{ch.KNOB_TAG_ID}"
