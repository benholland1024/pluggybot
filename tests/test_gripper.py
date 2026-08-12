"""Guards for the claw module (hub/gripper.py, milestone 8)."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.hub.coupling import (
  CLAW_JAW_Z, CLAW_PAD_HALF_H, CLAW_PENDANT_BOT, HUB_STATION_YS,
  module_power_contact,
)
from pluggybot.hub.gripper import GRIP_Z, ClawTool, CLAW_MODULE
from pluggybot.hub.swap import HubSwap


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def test_pendant_stops_at_the_jaw_tops():
  """The structure carrying a gripper must not occupy the space the gripper
  needs. A first version ran the drop tube 28 mm INTO the grip zone: it
  reached the block before the jaws did, shoved it 17 mm forward and 3 mm up,
  and the jaws then closed on empty air while the graze still registered as
  'both pads in contact'. Twice, because the constant did not follow when the
  pad height changed."""
  assert CLAW_PENDANT_BOT == pytest.approx(CLAW_JAW_Z + CLAW_PAD_HALF_H), \
    "the pendant must end exactly at the jaw tops, derived not re-typed"


def test_claw_reaches_the_floor_but_not_below(hub_model):
  """The whole design hangs on the pendant being long enough to reach an
  object on the floor with the lift near its bottom stop, and the pads being
  able to close without digging in."""
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[3])
  swap.pick()
  claw = ClawTool(hub_model, data, swap)
  claw.jaws(0.0)
  residual = claw.lower_grip_to(GRIP_Z)
  assert abs(residual) < 0.003, (
    f"could not put the jaws at floor level: {residual * 1000:.1f} mm out")
  assert float(claw.grip_world()[2]) - CLAW_PAD_HALF_H > -1e-6, \
    "the pads are being driven into the floor"


def test_claw_picks_powers_and_lifts(hub_model):
  """The milestone claim for the fourth tool: fetch it, power it through the
  coupling, aim it at something on the floor, grip and lift.

  Judged on CONTACT (both pads on the block) and on the block actually
  leaving the floor -- not on the jaw command, which happily reads 'closed'
  while gripping nothing.
  """
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[3])
  swap.pick()
  assert swap.module_state(CLAW_MODULE)["on_fork"], "claw pick failed"
  assert module_power_contact(hub_model, data, CLAW_MODULE), \
    "claw module is not powered through the coupling"

  claw = ClawTool(hub_model, data, swap)
  fwd, lat = claw.calibrate()
  assert 0.20 < fwd < 0.30 and -0.08 < lat < -0.02, \
    f"grip offset ({fwd:.3f}, {lat:.3f}) is not where the fork line is"

  obj = np.array(data.xpos[hub_model.body("pickup").id], dtype=float)
  z0 = float(obj[2])
  assert claw.drive_over((obj[0], obj[1]), 0.0), "never reached the block"
  # Aim is judged PER AXIS, because the jaws capture very differently in the
  # two: laterally the open jaws present a 62 mm gap, but fore-aft the pads
  # are only 28 mm deep. A single radial tolerance hid that, and it also had
  # to move when the arm was angled forward -- putting the grip 285 mm ahead
  # of the axle instead of 244 amplifies any heading error by 17 %. The real
  # criterion is the grasp below; this just localises a failure.
  g = claw.grip_world()
  lat, fwd = abs(float(g[1] - obj[1])), abs(float(g[0] - obj[0]))
  assert lat < 0.020, f"aim {lat * 1000:.1f} mm off laterally"
  assert fwd < 0.020, f"aim {fwd * 1000:.1f} mm off fore-aft (pads are 28 mm)"

  r = claw.pick_up()
  assert r["holding"], f"no grip: {r}"
  lifted = float(data.xpos[hub_model.body("pickup").id][2]) - z0
  assert lifted > 0.04, f"block only rose {lifted * 1000:.1f} mm"
  assert module_power_contact(hub_model, data, CLAW_MODULE), \
    "lifting the payload unseated the module"


def test_the_robot_cannot_see_what_it_grasps():
  """A design constraint, asserted so it cannot be quietly forgotten.

  The nav camera sits 180 mm up with a 41 deg fovy, so the floor leaves its
  view inside ~0.48 m, and the grip point is ~244 mm ahead of the axle. The
  object is therefore invisible at the moment of grasping -- the same shape
  as the socket vanishing from the dock camera at close range. Any real
  pickup must memorise the pose while the target is still visible and then
  run open-loop.
  """
  cam_z, fovy = 0.180, 41.0
  blind_within = cam_z / math.tan(math.radians(fovy / 2))
  assert blind_within > 0.24, (
    "if the camera could see the floor at the grip point, the pickup could "
    "be closed-loop and this constraint would no longer hold")
  assert blind_within == pytest.approx(0.48, abs=0.02)
