"""Guards for the fork robot + hub world integration (milestone 8)."""

import mujoco
import pytest

from pluggybot.hub.coupling import HUB_STATION_YS
from pluggybot.hub.swap import HubSwap

# Adjacent-link clearance for the FORK, the same geometric discipline as
# test_arm.py: weld/parent filtering silences these contacts, so overlap is
# asserted from geometry through the whole lift+arm envelope.
FORK_GEOMS = ("fork_bridge", "fork_prong_l", "fork_prong_r",
              "fork_vl_a", "fork_vl_b", "fork_vr_a", "fork_vr_b")
CLEARANCE_REFS = ("head_mount", "mast", "lift_motor", "chassis", "battery")


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def _world_half_extents(model, data, gid):
  """Box half-extents projected onto world axes (exact for axis-aligned
  geoms, a tight conservative bound for the 45-degree V plates -- unlike a
  bounding sphere, which inflates the wide flat bridge into a 65 mm ball)."""
  size = model.geom(gid).size
  xmat = data.geom_xmat[gid].reshape(3, 3)
  return [sum(abs(float(xmat[i][j])) * float(size[j]) for j in range(3))
          for i in range(3)]


def _aabb_overlap(model, data, a, b):
  ga, gb = model.geom(a).id, model.geom(b).id
  pa, pb = data.geom_xpos[ga], data.geom_xpos[gb]
  ha, hb = _world_half_extents(model, data, ga), _world_half_extents(model, data, gb)
  return all(abs(float(pa[k]) - float(pb[k])) < ha[k] + hb[k] for k in range(3))


def test_fork_stows_below_the_scan_row(hub_model):
  d = mujoco.MjData(hub_model)
  d.qpos[0] = 2.0                     # far from the hub
  mujoco.mj_forward(hub_model, d)
  for _ in range(1000):
    mujoco.mj_step(hub_model, d)
  for g in FORK_GEOMS:
    gid = hub_model.geom(g).id
    top = float(d.geom_xpos[gid][2]) + _world_half_extents(hub_model, d, gid)[2]
    assert top < 0.18, f"{g} reaches the scan row while parked (top {top:.3f})"


def test_fork_transit_clears_the_robot(hub_model):
  d = mujoco.MjData(hub_model)
  d.qpos[0] = 2.0
  mujoco.mj_forward(hub_model, d)
  for _ in range(500):
    mujoco.mj_step(hub_model, d)
  lift = hub_model.actuator("lift").id
  arm = hub_model.actuator("arm").id
  hits = set()
  for step in range(3000):
    d.ctrl[lift] = 0.31 * step / 3000
    mujoco.mj_step(hub_model, d)
    if step % 20 == 0:
      hits |= {(a, b) for a in FORK_GEOMS for b in CLEARANCE_REFS
               if _aabb_overlap(hub_model, d, a, b)}
  for step in range(1500):
    d.ctrl[arm] = 0.20 * step / 1500
    mujoco.mj_step(hub_model, d)
    if step % 20 == 0:
      hits |= {(a, b) for a in FORK_GEOMS for b in CLEARANCE_REFS
               if _aabb_overlap(hub_model, d, a, b)}
  assert not hits, f"fork sweeps through the robot: {sorted(hits)}"


def test_robot_swap_cycle(hub_model):
  """The milestone-8 integration claim: the real base + lift + odometry runs
  the coupling verbs end to end -- pick the LCD module off the rack, carry it
  away, bring it back, hang it up. And the freestanding rack must not move:
  stability is measured, not assumed (it is a free body leaning on the wall
  through its braces; without them a sustained press scooted it 9.6 mm)."""
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap.pick()
  st = swap.module_state("module_lcd")
  assert st["on_fork"], f"pick failed: module at {st['pos']}"
  assert st["pos"][0] > 0.15, "module did not leave the rack"
  swap.put_back()
  st = swap.module_state("module_lcd")
  assert st["hung"], f"return failed: module at {st['pos']}"
  rack = data.xpos[hub_model.body("rack").id]
  assert abs(float(rack[0])) < 0.005 and abs(float(rack[1])) < 0.005, \
    "the rack moved during the swap"


def test_charge_bay_press_connects(hub_model):
  """Nosing into the charge bay and pressing must land BOTH pogo pins on the
  bumper -- the rack-side charge criterion -- without shoving the rack."""
  import math
  from pluggybot.hub.coupling import CHARGE_BAY_Y, rack_charge_contact
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  yaw = math.pi
  data.qpos[0] = 0.60 + 0.08 * math.cos(yaw)
  data.qpos[1] = CHARGE_BAY_Y
  data.qpos[2] = 0.045
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(hub_model, data)
  swap.reckoner.x, swap.reckoner.y, swap.reckoner.theta = 0.60, CHARGE_BAY_Y, yaw
  swap._drive_until(0.32, 0.04, timeout=15.0)
  assert rack_charge_contact(hub_model, data), "pins not both on the bumper"
  rack = data.xpos[hub_model.body("rack").id]
  assert abs(float(rack[0])) < 0.005, "the press shoved the rack"
