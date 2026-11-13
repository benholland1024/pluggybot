"""A foot on a lab plate (issue #403): the cage logs a press as the robot
whose foot is on the pad, and a press no errand of that plate made is the
robot's own `press` event, said with what it was doing. The lab's paid
jobs on legs, their rule and the plate act built off a sign went with the
quadruped (#559); the humanoids are shown the lab and press its plates
only by walking over them. Each rule pinned as cheaply as it fails for the
right reason."""

from collections import deque
from types import SimpleNamespace

import mujoco
import pytest

from pluggybot.activity import cage as cg
from pluggybot.lifecycle import world_facts
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import lang
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


# ---- a press, and whose it was -----------------------------------------------------


def _plate_world(on_pad: str = "pluggybot", at: dict | None = None):
  """A floor, the lab's cage and plates at the origin, one robot (a 15 kg
  block, over the plate's `LOAD_ON_N`) standing in the feed plate's corner
  and the other beside the plate, NEARER its centre and the cage's, touching
  nothing of it: the plate is the floor's square, so whose it is is where
  its contacts are. `at` places them otherwise, {root: (x, y, z, kg)}."""
  body, sensors = cg.cage_xml((0.0, 1.2))
  if at is None:
    at = {on_pad: (0.175, 0.175, 0.03, 15)}
    at[next(r for r in ("pluggybot", "r2_pluggybot") if r != on_pad)] = (0.0, 0.225, 0.03, 15)
  robots = "".join(f"""
    <body name="{name}" pos="{x} {y} {z}"><freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="{kg}"/></body>"""
                   for name, (x, y, z, kg) in sorted(at.items()))
  xml = f"""<mujoco><option timestep="0.002"/><worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>{body}{robots}
  </worldbody><sensor>{sensors}</sensor></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  return model, mujoco.MjData(model)


@pytest.mark.parametrize("on_pad", ["pluggybot", "r2_pluggybot"])
def test_the_cage_logs_a_press_as_the_robot_whose_foot_is_on_the_pad(on_pad):
  model, data = _plate_world(on_pad)
  cage = cg.Cage(model, data)
  for _ in range(1500):
    mujoco.mj_step(model, data)
    cage.sense(model, data)
    if cage.presses:
      break
  [press] = list(cage.presses)
  assert press["plate"] == "feed" and press["robot"] == on_pad, press
  assert press["before"] == "resting" and press["after"] == "eating"
  assert cage.press_seq == 1 and cage.counts["feed"] == 1


def test_two_robots_on_one_plate_its_press_is_the_one_whose_weight_pressed_it():
  """Both on the feed plate, the first robot in the model's order a 2 kg
  block and the second 15 kg: the press is the second's, whose weight the
  load cell read past `LOAD_ON_N`, never the first robot with a contact
  there (#684: a plate reads its load)."""
  model, data = _plate_world(at={"pluggybot": (0.1, 0.1, 0.03, 2),
                                 "r2_pluggybot": (-0.1, -0.1, 0.03, 15)})
  cage = cg.Cage(model, data)
  for _ in range(1500):
    mujoco.mj_step(model, data)
    cage.sense(model, data)
    if cage.presses:
      break
  [press] = list(cage.presses)
  assert (press["plate"], press["robot"]) == ("feed", "r2_pluggybot"), press


def _press(seq, plate, robot="pluggybot"):
  return {"seq": seq, "t": float(seq), "plate": plate, "robot": robot,
          "before": "resting", "after": "on_its_side" if plate == "shock" else "eating"}


def test_a_press_no_errand_of_that_plate_made_is_its_own_event():
  """Outside any errand, every press of this robot's is a `press` event
  with what it was doing, and inside its own procedure the procedure's
  name; another robot's presses are its own lifecycle's to say."""
  life = stub_life()
  events: list = []
  life.on_event.append(events.append)
  fake = SimpleNamespace(presses=deque(), press_seq=0)
  life._press_of, life._press_cage = life.activities, fake

  def press(*presses):
    fake.presses.extend(presses)
    fake.press_seq = presses[-1]["seq"]
    life._press_step()
    return [e for e in events if e["type"] == "press"]

  seen = press(_press(1, "shock"), _press(2, "feed", robot="r2_pluggybot"))
  assert [(e["plate"], e["doing"]) for e in seen] == [("shock", life.state.lower())]
  walk = lang.compile_procedure("def walk():\n  drive_to(25.0, 3.0)\n",
                                world_facts("home_humanoid"))
  life._errand_now = programmed_errand(walk, task="program", name="procedure")
  seen = press(_press(3, "feed"))
  assert seen[-1]["plate"] == "feed" and seen[-1]["doing"] == "procedure:walk"
  assert seen[-1]["before"] == "resting" and seen[-1]["robot"] == life.root
  # ...and a seq already seen is never said twice
  n = len(seen)
  life._press_step()
  assert len([e for e in events if e["type"] == "press"]) == n


def test_the_press_hook_is_on_the_seam_and_finds_the_worlds_cage():
  """The one line of wiring: the lifecycle's own step hook reads the cage
  in the world's activities -- no hand-set cache -- and says the press."""
  life = stub_life()
  events: list = []
  life.on_event.append(events.append)
  model, data = _plate_world()
  cage = cg.Cage(model, data)
  life.activities = [cage]
  cage.presses.append(_press(1, "shock"))
  cage.press_seq = 1
  life.body.run(life.body.hold_routine(0.05))
  [press] = [e for e in events if e["type"] == "press"]
  assert press["plate"] == "shock" and press["robot"] == life.root
