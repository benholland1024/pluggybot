"""Guards for the seed dispenser (tools/dispenser.py, the fifth tool).

The first tool built against `docs/ToolPattern.md`, so these follow the
checklist that doc's "pytest regressions" section lays out: the tool can be
fetched and is powered; the working point lands where it should; the job
itself succeeds on a PHYSICAL criterion; the module is still seated
afterwards; derived constants stay derived; and the one measured surprise of
the build gets an assertion that can fail.
"""

import mujoco
import numpy as np
import pytest

from pluggybot.rack.coupling import (
  DISP_BORE, DISP_MOUTH_Z, DISP_POCKET_HALF_Z, DISP_SHELF_END,
  DISP_SHELF_TOP, DISP_STROKE, HUB_STATION_YS, SEED_COUNT, SEED_R,
  module_power_contact, seed_stack_zs,
)
from pluggybot.tools.dispenser import (
  SEED_MODULE, SOW_OUTLET_Z, SeedDispenser,
)
from pluggybot.rack.swap import HubSwap


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


@pytest.fixture
def picked(hub_model):
  """The dispenser on the fork, powered, calibrated.

  FUNCTION-scoped, deliberately, even though the fetch costs ~5 s each time.
  A module-scoped version passed the whole file and then failed when a
  single test was run alone: dispensing empties the magazine, so the landing
  test was only ever passing on the seeds an EARLIER test had dropped. A
  fixture that carries mutated state between tests is a shared world
  pretending to be a fresh one.
  """
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[4])
  swap.pick()
  disp = SeedDispenser(hub_model, data, swap)
  disp.calibrate()
  return hub_model, data, swap, disp


# ---- geometry: the constants that must stay derived -------------------------

def test_the_escapement_stroke_clears_the_bore():
  """The blanking slab must fully cover the magazine mouth at the out
  position, and the pocket must fully clear the shelf's end at the SAME
  moment -- that simultaneity is the whole mechanism. A stroke shorter than
  the bore leaves a gap the stack pours through; one shorter than the shelf
  overhang drops nothing at all."""
  assert DISP_STROKE > 2 * DISP_BORE, \
    "stroke does not cover the bore: the stack will pour out"
  assert DISP_STROKE - DISP_BORE > DISP_SHELF_END, \
    "the pocket never fully clears the shelf end: nothing will drop"


def test_the_pocket_is_deeper_than_a_seed():
  """A metered seed rides in the pocket while the module swings ~10 deg in
  transit. A pocket shallower than the seed is a ramp, not a pocket."""
  assert DISP_POCKET_HALF_Z * 2 > 2 * SEED_R, \
    "the pocket cannot cage a seed"


def test_the_magazine_is_loaded_below_the_plate_and_above_the_shelf():
  """The whole tool hangs BELOW the module plate, in air the rack never
  occupies -- the pattern's advice after the pen module (which stands 26 mm
  proud AT plate height) turned into the open stow failure. Loading a seed
  outside the tube would put it through a wall on the first step."""
  zs = seed_stack_zs()
  assert len(zs) == SEED_COUNT
  assert zs[0] > DISP_SHELF_TOP, "the first seed starts below its own shelf"
  assert zs[-1] + SEED_R < -0.030, "the stack does not fit under the plate"
  assert all(b - a > 2 * SEED_R * 0.9 for a, b in zip(zs, zs[1:])), \
    "the loaded stack starts interpenetrating"


# ---- the fetch --------------------------------------------------------------

def test_dispenser_picks_and_powers(picked):
  """The milestone claim for any tool: fetch it from its bay and have it
  powered through the coupling. Judged ELECTRICALLY, because `on_fork` is a
  3-4 cm position heuristic that has reported True right through a module
  being ejected."""
  model, data, swap, _ = picked
  assert swap.module_state(SEED_MODULE)["on_fork"], "dispenser pick failed"
  assert module_power_contact(model, data, SEED_MODULE), \
    "the dispenser is not powered through the coupling"


def test_outlet_offset_is_on_the_fork_line(picked):
  """Measured, not derived -- and the assertion is that the MEASUREMENT lands
  where the fork line is. The pen's tip was 72 mm below the peg against a
  47 mm estimate and the claw's grip was 12.5 mm out along track, both
  because nominal geometry contains neither droop nor the module's lean."""
  _, _, _, disp = picked
  fwd, lat = disp.offset
  assert 0.18 < fwd < 0.30, f"outlet {fwd:.3f} m ahead of the axle is not the fork"
  assert -0.10 < lat < -0.02, f"outlet lateral {lat:.3f} m is not the fork line"


# ---- the job ----------------------------------------------------------------

def test_escapement_meters_exactly_one_seed_per_cycle(picked):
  """THE claim of the tool, and the reason the mechanism is an escapement
  rather than a gate: the count comes from geometry, so it cannot drift with
  timing or load. Judged on seeds that actually LEFT the tool, not on the
  gate command.

  Three cycles, because one cycle cannot distinguish "meters one" from
  "dumps whatever is nearest the exit".
  """
  model, data, swap, disp = picked
  assert disp.remaining() == SEED_COUNT, "magazine did not start full"
  for cycle in range(SEED_COUNT):
    r = disp.dispense()
    assert r["count"] == 1, (
      f"cycle {cycle} metered {r['count']} seeds, not 1")
    assert disp.remaining() == SEED_COUNT - 1 - cycle
  assert disp.remaining() == 0
  assert module_power_contact(model, data, SEED_MODULE), \
    "dispensing unseated the module"


def test_a_dropped_seed_stops_where_it_lands(hub_model):
  """The build's one real surprise, and it cost 220 mm of sowing accuracy.

  A sphere on a plane with MuJoCo's default `condim=3` has NO rolling
  resistance and rolls until something stops it -- and sliding friction does
  not touch it, because a rolling ball is not sliding. Measured on a seed
  released with 0.15 m/s of residual motion: 586 mm travelled at mu=0.7,
  and 586 mm at mu=1.0 with priority, identical to the millimetre. Only
  `condim="6"` with a rolling term brings it to 14 mm.

  This test drops a seed from the sow height with that same residual motion
  and fails if it wanders. Removing the seed's condim/rolling friction in
  `coupling.SEED_FRICTION` makes it fail by a factor of 40.
  """
  data = mujoco.MjData(hub_model)
  bid = hub_model.body("seed_0").id
  adr = hub_model.joint(hub_model.body_jntadr[bid]).qposadr[0]
  dof = hub_model.body_dofadr[bid]
  # Teleport seed 0 to open floor at sow height, moving, and let it go.
  data.qpos[adr:adr + 3] = [1.60, -0.60, SOW_OUTLET_Z]
  data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
  mujoco.mj_forward(hub_model, data)
  data.qvel[dof] = 0.15
  start = np.array(data.qpos[adr:adr + 2], dtype=float)
  for _ in range(6000):                       # 6 s
    mujoco.mj_step(hub_model, data)
  travelled = float(np.linalg.norm(
    np.array(data.qpos[adr:adr + 2], dtype=float) - start))
  assert travelled < 0.10, (
    f"a sown seed rolled {travelled * 1000:.0f} mm -- rolling resistance is "
    "gone (condim=3 spheres roll forever, and sliding friction will not "
    "stop them)")
  assert float(np.linalg.norm(data.qvel[dof:dof + 3])) < 0.01, \
    "the seed never came to rest"


def test_sowing_at_a_point_lands_a_seed_there(picked):
  """The errand end to end, judged the way a gardener would: drive to a
  spot, meter one seed, and have it come to rest ON THE GROUND near the
  spot. Contact with the floor is the fact; the gate command is the belief.

  The tolerance is CENTIMETRE class on purpose. That is this tool's design
  target, not a shortfall -- a sown seed lands where it lands, and matching
  the control effort to the tolerance class is why this controller skips the
  claw's back-up-and-retry refinement entirely. It is still a real bound: it
  fails by 4x if the seeds lose their rolling resistance.
  """
  model, data, swap, disp = picked
  target = (0.95, 0.55)
  disp.drive_over(target, 0.0)
  disp.lower_outlet_to(SOW_OUTLET_Z)
  r = disp.dispense()
  assert r["count"] == 1, f"metered {r['count']} seeds at the sow point"
  k = r["dropped"][0]
  swap._run(2.0, 0.0)                       # let it come to rest
  assert disp.landed(k), "the sown seed never reached the ground"
  p = data.xpos[disp.seed_bids[k]]
  err = float(np.hypot(float(p[0]) - target[0], float(p[1]) - target[1]))
  assert err < 0.06, f"seed landed {err * 1000:.0f} mm from the target"
  assert module_power_contact(model, data, SEED_MODULE), \
    "sowing unseated the module"


def test_the_dispenser_spends_none_of_the_coupling_moment_budget(hub_model):
  """A design property, asserted so it cannot be quietly lost.

  The gravity latch takes ~0.45 N.m of pitch moment, and a payload at forward
  offset L costs W*L -- which is why the claw is a pendant rather than an
  arm. The dispenser's magazine hangs on the peg's OWN axis, so it spends
  nothing however much it carries. Moving the tube forward for packaging
  reasons would silently start spending that budget.
  """
  outlet = hub_model.site("seed_outlet")
  assert abs(float(outlet.pos[0])) < 0.005, (
    "the outlet has moved off the peg axis -- the magazine now costs pitch "
    "moment against the coupling's 0.45 N.m budget")
  assert DISP_MOUTH_Z < -0.030, \
    "the magazine no longer hangs clear below the module plate"
