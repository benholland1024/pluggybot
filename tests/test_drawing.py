"""Guards for the drawing tool's plot controller (hub/drawing.py, milestone 8)."""

import math

import mujoco
import pytest

from pluggybot.hub import strokes
from pluggybot.hub.coupling import HUB_STATION_YS, module_power_contact
from pluggybot.hub.drawing import (
  PEN_MODULE, PenPlotter, circle_path, pen_on_board, square_path,
)
from pluggybot.hub.swap import HubSwap


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def test_paths_are_closed_and_sized():
  for path in (square_path(0.075), circle_path(0.075)):
    assert math.dist(path[0], path[-1]) < 1e-9, "figure must close"
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    assert 0.070 < max(xs) - min(xs) <= 0.0751
    assert 0.070 < max(ys) - min(ys) <= 0.0751


def test_board_is_static(hub_model):
  """A board that slides under the pen would be measuring the board, not the
  plotter -- so it must have no freejoint."""
  gid = hub_model.geom("board").id
  bid = hub_model.geom_bodyid[gid]
  assert bid == 0, "the board must be welded to the world"


def test_squaring_up_does_not_overshoot(hub_model):
  """_face's P-controller stops commanding when the error goes small, but
  `slew` rate-limits the wheel command, so the robot coasts past: measured
  -9.5 deg in, +7.5 deg out. Squareness is load-bearing here -- the carriage
  sweeps 110 mm across the board, so a yaw error theta swings pen depth by
  110*sin(theta), and 7.5 deg is 14 mm, most of the quill's whole travel.
  """
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  plotter = PenPlotter(hub_model, data, swap)
  assert plotter.drive_to_board(), "never reached the board"
  w, x, y, z = data.qpos[3:7]
  true_yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  assert abs(true_yaw) < 2.0, (
    f"parked {true_yaw:.2f} deg off square to the board -- that is "
    f"{110 * math.sin(math.radians(abs(true_yaw))):.1f} mm of pen-depth swing "
    f"across the carriage stroke")


@pytest.mark.slow
def test_pen_module_draws_a_figure(hub_model):
  """The milestone claim: the robot fetches an axis it does not own and uses
  it. Fetch the pen module, carry it to the board, and plot a circle.

  Judged on SHAPE error -- distance from the commanded figure -- not on
  same-instant tracking error, because a figure traced slightly behind
  schedule is still the right figure on the board.
  """
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  assert module_power_contact(hub_model, data, PEN_MODULE), \
    "the pen module is not powered through the coupling"

  plotter = PenPlotter(hub_model, data, swap)
  assert plotter.drive_to_board(), "never reached the board"
  r = plotter.draw(circle_path())

  assert r["drew"], f"never got the pen on the board: {r}"
  assert r["inked_fraction"] > 0.85, (
    f"pen was only on the board for {r['inked_fraction']:.0%} of the path")
  # FORM is the mechanics claim: is it the right shape? A rigid offset is a
  # calibration constant and is asserted separately and loosely -- a square
  # drawn perfectly but 17 mm to one side is a very different (and much more
  # fixable) machine than one that draws a wobbly square in the right place.
  assert r["form_rms_mm"] < 4.0, (
    f"figure is the wrong SHAPE: {r['form_rms_mm']:.1f} mm rms after "
    f"removing a {r['offset_mm']:.1f} mm rigid offset")
  assert r["offset_mm"] < 15.0, (
    f"figure landed {r['offset_mm']:.1f} mm from where it was asked for")
  assert module_power_contact(hub_model, data, PEN_MODULE), \
    "the tool came unseated while drawing"
  assert not pen_on_board(hub_model, data), \
    "the pen was left down on the board after finishing the figure"


@pytest.mark.slow
def test_multi_stroke_program_lifts_between_strokes(hub_model):
  """The one physical claim stroke programs add (issue #11): between two
  polylines the pen comes OFF the board.

  A figure of one closed loop -- which is both diagnostics and every figure
  the plotter had drawn until now -- cannot test this, because there is
  nowhere for the pen to travel to. Get it wrong and the drawing gains a
  line straight through itself from the end of each stroke to the start of
  the next, while `inked_fraction` goes UP and the shape error barely moves:
  the fault flatters every number already being watched. Hence the travel
  rows in `trace`, and hence checking that they exist before believing the
  verdict computed from them.

  The second claim, and the one that took measuring: every stroke re-presses,
  and each press seats the module slightly differently, so the strokes land in
  DIFFERENT PLACES. Measured on this exact word before the per-stroke re-zero:
  lateral offsets ran from -4.55 mm on the 'G' to +2.78 mm on the 'L', a
  7.3 mm spread across a 25 mm-high word -- letters whose parts miss each
  other, with the G's crossbar floating clear of its arc. Re-zeroing each
  stroke against the first press's bias took that to 4.7 mm and the form error
  from 1.91 to 1.10 mm. Both thresholds below fail on the un-re-zeroed run.

  "PLUG" rather than a single letter because the effect is POSITION dependent:
  it only shows when strokes sit at opposite ends of the carriage's travel.
  """
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  plotter = PenPlotter(hub_model, data, swap)
  assert plotter.drive_to_board(), "never reached the board"

  program = strokes.program("text", text="PLUG", cap_height=0.025)
  assert len(program.strokes) == 7
  r = plotter.draw_program(program)

  assert r["drew"], f"never got the pen on the board: {r}"
  assert r["strokes_drawn"] == 7, (
    f"only {r['strokes_drawn']} of 7 strokes found the board -- a stroke that "
    f"misses is a gap in a letter")
  travel = [t for t in plotter.trace if t[6] < 0]
  assert travel, ("no inter-stroke travel was recorded, so the pen-lift check "
                  "below is vacuous")
  assert r["travel_ink_fraction"] == 0.0, (
    f"the pen inked on {r['travel_ink_fraction']:.0%} of the travel between "
    f"strokes -- the word has lines through it that nobody drew")
  assert r["inked_fraction"] > 0.85, (
    f"pen was only down for {r['inked_fraction']:.0%} of the strokes")
  assert r["form_rms_mm"] < 1.5, (
    f"word is the wrong SHAPE: {r['form_rms_mm']:.2f} mm rms "
    f"(1.91 mm without the per-stroke re-zero, 1.10 with)")

  offsets = []
  for i in range(len(program.strokes)):
    ink = [t for t in plotter.trace if t[5] and t[6] == i]
    offsets.append(sum(t[1] - t[3] for t in ink) / len(ink))
  spread = (max(offsets) - min(offsets)) * 1000
  assert spread < 6.0, (
    f"strokes landed {spread:.1f} mm apart laterally (measured 7.3 mm without "
    f"the re-zero, 4.7 with) -- the parts of a letter miss each other")
  assert module_power_contact(hub_model, data, PEN_MODULE), \
    "the tool came unseated while drawing"


@pytest.mark.slow
def test_loaded_calibration_moves_the_origin(hub_model):
  """What the free-air calibration gets wrong is the ORIGIN, not the scale.

  Pressing the pen deflects the module and the compliant wrist, moving the
  pen's home position ~10 mm sideways; re-zeroing there took a drawn circle
  from 9.2 mm to 2.2 mm of shape error. The gain, measured at rest, barely
  moves -- so this test asserts the offset, which is the effect that is
  really there, rather than the scale change I first assumed and which the
  measurement did not support.
  """
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  plotter = PenPlotter(hub_model, data, swap)
  assert plotter.drive_to_board()
  free = dict(plotter.calibrate())
  loaded = dict(plotter.calibrate_loaded(0.0375))
  assert abs(free["dy_dcarriage"]) > 0.95, \
    f"free-air gain should be ~1, got {free['dy_dcarriage']:.3f}"
  assert loaded["loaded"] is True
  shift = abs(loaded["y0"] - free["y0"])
  # The original 10 mm shift was measured before the wheels had their
  # parking brake (issue #3): part of "pressing displaces the pen" was the
  # whole BASE rolling under the press. With the base held, the remaining
  # ~4 mm is the genuine module + compliant-wrist deflection -- still real,
  # still worth re-zeroing for, just no longer inflated by the robot
  # quietly driving.
  assert shift > 0.0025, (
    f"loaded calibration only moved the origin {shift * 1000:.1f} mm -- if "
    f"pressing does not displace the pen, this pass is dead weight")
