"""The first paid job on legs (issue #403): the lab back in the quadruped's
world, its rule true there and the rover's byte for byte, `feed_mouse`
offered alone, the plate found by its sign and pressed off it (#419's places), and a
press no errand of that plate made recorded as its own event. Each rule
pinned as cheaply as it fails for the right reason; the job flown whole on
the pair is ladder A's, by hand (`scripts/solve.py --feature mouse --pair
--body quadruped`)."""

import hashlib
import json
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot.activity import cage as cg
from pluggybot.economy import cadence as cad
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.scoring import default_table
from pluggybot.lifecycle import QUAD_HOME, world_config
from pluggybot.mind import overseer as ov
from pluggybot.mind.thoughts import ThoughtFiles
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


def _quad_mind(autonomous: bool = True, **kw):
  return ov.build(QUAD_HOME, None, enabled=True, client=object(), ledger=Ledger(),
                  thoughts=ThoughtFiles.open(None, body="quadruped"),
                  robot_name="Luca", mortal=True, hearts=True,
                  autonomous=autonomous, origin="unseeded" if autonomous else "none",
                  standing_orders=autonomous, others=("Rowan",), **kw)


def _section(boss, heading: str) -> str:
  return next(text for name, text in boss.sections if name == heading)


# ---- the lab in the quadruped's world ---------------------------------------------


def test_the_lab_is_in_the_quadrupeds_world_with_its_rule_disclosure_and_care():
  """The house's lab, its props where the rover's were; on `autonomous` the
  zone's whole grammar -- `care`, `real`, `mouse_will`, the rule with the
  disclosure line once -- and on `guarded` none of it."""
  assert world_config(QUAD_HOME)["lab"] == world_config("home")["lab"]
  boss = _quad_mind()
  assert boss.menu.lab == "lab" and "care" in boss.menu.available()
  assert boss.menu.lab_jobs == ("feed_mouse",) and boss.menu.lab_route is False
  text = "\n".join(b["text"] for b in boss.system)
  assert text.count(ov.DISCLOSURE) == 1
  props = boss.menu.schema()["properties"]
  assert {"care", "real", "mouse_will"} <= set(props)
  guarded = _quad_mind(autonomous=False)
  assert guarded.menu.lab == "" and ov.DISCLOSURE not in guarded.system[0]["text"]
  # ...and the context's `lab` block reads no road there: none is surveyed
  assert lc.lab_route(QUAD_HOME) == [] and lc.lab_route("home")


def test_the_rule_on_legs_is_in_its_own_words_and_names_only_the_job_it_offers():
  rule = _section(_quad_mind(), "THE LAB")
  assert "when a foot presses it" in rule and "walk onto the feed plate" in rule
  assert "it costs the walk." in rule
  for gone in ("wheel", "drive", "`shock_mouse`", "not otherwise", "`route`"):
    assert gone not in rule, gone
  [job] = [ln for ln in rule.splitlines() if "`feed_mouse`" in ln]
  assert job.startswith("- One of the plates is also pressed on a JOB")
  assert "The job asks for `mouse_will` first" in job
  # the plate still does what its name says: the world did not change
  assert "`shock`, `feed` and `toy`. Each does what its name says" in rule


#: `lab_rule("lab")` and `lab_rule("lab", decline=False)` as the rover's
#: world has read them since #287. Moved on purpose or not at all.
ROVER_LAB_SHA = {True: "1b2d77dbaadeb068a985a57bd465605cfe60ecfb5a20f72623dd5a4afea730d3",
                 False: "1f068ac3e9ca116737c2454f2a79760bcfb0a9a39525b6d679647fb8e8304f18"}


def test_a_world_with_both_jobs_reads_the_rule_byte_for_byte():
  for decline, sha in ROVER_LAB_SHA.items():
    assert hashlib.sha256(ov.lab_rule("lab", decline=decline).encode()).hexdigest() == sha
  rover = ov.build("home", lc.board_book("home"), enabled=True, client=object(),
                   thoughts=ThoughtFiles(), autonomous=True)
  assert rover.menu.lab_jobs == ("shock_mouse", "feed_mouse") and rover.menu.lab_route
  assert _section(rover, "THE LAB") == ov.lab_rule("lab")
  assert ov.lab_rule("lab", jobs=("feed_mouse", "shock_mouse")) == ov.lab_rule("lab")


def test_the_jobs_bullet_says_what_is_offered_and_nothing_else():
  shock = ov.lab_rule("lab", jobs=("shock_mouse",))
  assert "`feed_mouse`" not in shock and "not otherwise" in shock
  none = ov.lab_rule("lab", jobs=())
  assert "JOB" not in none and "`mouse_will`" not in none
  # the free feed's own line names the paid one only where it is offered
  menu = ov.Menu.for_world(QUAD_HOME)
  for jobs, says in (((), False), (("feed_mouse",), True), (None, True)):
    care = ov.system_prompt(ThoughtFiles(), replace(menu, lab="lab", lab_jobs=jobs),
                            default_table(), autonomous=True, lab="lab")[0]["text"]
    [line] = [ln for ln in care.splitlines() if ln.strip().startswith('"care":')]
    assert ("`feed_mouse`" in line) is says, (jobs, line)


# ---- the offers -----------------------------------------------------------------


def test_home_quad_offers_feed_mouse_alone_and_the_shock_is_one_line_back(tmp_path):
  """A test reads the entry: `feed_mouse` and nothing else; putting the
  shock job back is one line in `kinds`, and the prompt it builds then
  names both jobs as the rover's world does."""
  shipped = cad.Cadence.load(QUAD_HOME, cad.CADENCE_PATH)
  assert list(shipped.kinds) == ["feed_mouse"]
  doc = json.loads(cad.CADENCE_PATH.read_text())
  block = doc["worlds"][QUAD_HOME]
  block["kinds"] = {"shock_mouse": {}, **block["kinds"]}           # the one line
  path = tmp_path / "cadence.json"
  path.write_text(json.dumps(doc))
  back = cad.Cadence.load(QUAD_HOME, path)
  assert list(back.kinds) == ["shock_mouse", "feed_mouse"]
  # ...and the producer rotates the lab's jobs alone: the cage is the one
  # target this world names for any of them
  targets = lc.world_targets(QUAD_HOME, procedures=True, robots=("Luca", "Rowan"))
  assert targets["cage"] == ["lab"]
  from pluggybot.economy.tasks import TaskBoard
  producer = lc.task_producer(TaskBoard(), QUAD_HOME, cadence=shipped,
                              procedures=True, robots=("Luca", "Rowan"))
  assert producer.kinds == ("feed_mouse",)
  # ...priced off a measured row, never the table's unpriced fallback
  from pluggybot.economy import energy
  assert energy.load(QUAD_HOME).errand_wh.get("feed", 0.0) > 0.0


# ---- the walk: found by its sign, pressed off it (#419's places) ------------


def test_on_legs_a_plate_act_is_found_by_its_sign_and_pressed_from_the_labs_address():
  """No position finer than the house (#419): a plate act on legs is `find`
  round the lab's ADDRESS and `press` off the plate's own sign -- the only
  numbers in it are the address's -- and the find is the way there, so a
  job that never found its plate never reached the cage. It validates
  against the world's own verbs. The rover keeps its surveyed road."""
  from pluggybot.home.places import area
  from pluggybot.procedure.steps import compile_program
  at = area("lab")["address"]
  dock = tuple(world_config(QUAD_HOME)["dock"][:2])
  for act in ("feed", "toy", "shock"):
    steps = lc.cage_program(QUAD_HOME, act, dock).steps()
    tag = cg.PLATE_TAGS[act]
    assert [(s.verb, dict(s.args)) for s in steps] == [
      ("find", {"tag": tag, "x": at["x"], "y": at["y"]}), ("press", {"tag": tag})], act
    compile_program(lc.cage_program(QUAD_HOME, act), lc.world_facts(QUAD_HOME))
    assert lc.cage_errand(QUAD_HOME, act, from_xy=dock).detail["routeLegs"] == 1
  with pytest.raises(ValueError):
    lc.cage_program(QUAD_HOME, "company")
  rover = lc.cage_program("home", "feed", (0.5, -1.0)).steps()
  assert {s.verb for s in rover} == {"drive_to"}, "the rover keeps its road"


def test_on_legs_care_is_the_plates_alone():
  """Company is a spot beside the cage no tag marks: code would have to
  hand it over, so on legs `care` is `feed` or `toy` -- in the grammar,
  the rule and the action's line -- and the rule still says standing
  beside the cage is company, which the robot may walk to on its own."""
  boss = _quad_mind()
  assert boss.menu.care_acts == cg.PLATE_CARE_ACTS == ("feed", "toy")
  assert [a for a in boss.menu.schema()["properties"]["care"]["enum"] if a] == ["feed", "toy"]
  rule = _section(boss, "THE LAB")
  assert "`feed` (walk onto the feed plate) or `toy` (the toy plate)." in rule
  assert "`company` (stand" not in rule and "Standing beside the cage is company." in rule
  text = "\n".join(b["text"] for b in boss.system)
  [line] = [ln for ln in text.splitlines() if ln.strip().startswith('"care":')]
  assert "`care` names `feed` or `toy`." in line
  assert lc.errand_from(ov.Decision(action="care", care="company"), QUAD_HOME) is None
  rover = ov.build("home", lc.board_book("home"), enabled=True, client=object(),
                   thoughts=ThoughtFiles(), autonomous=True)
  assert rover.menu.care_acts == ("feed", "toy", "company")


def test_a_feed_job_on_legs_finds_then_presses_the_feed_plate():
  """The wiring, on a stub body: the job's errand asks the body to find the
  feed plate's tag and then press it -- and, where it has seen the plate,
  the press is made."""
  life = stub_life(QUAD_HOME)
  feed = cg.PLATE_TAGS["feed"]
  life.body.places.see(feed, 25.0, 4.8, 0.0, view=-1.57)
  life.run_errand(lc.cage_errand(QUAD_HOME, "feed", task="feed"))
  assert life.body.found == [feed] and life.body.pressed == [feed]


# ---- a press no errand of that plate made ------------------------------------------


def _plate_world(on_pad: str = "pluggybot"):
  """A floor, the lab's cage and plates at the origin, one robot (a 1.5 kg
  block) standing on the feed pad's corner and the other beside the pad
  and NEARER its centre, touching nothing of it."""
  body, sensors = cg.cage_xml((0.0, 1.2))
  at = {on_pad: (0.18, 0.18, 0.08)}
  at[next(r for r in ("pluggybot", "r2_pluggybot") if r != on_pad)] = (0.25, 0.0, 0.04)
  robots = "".join(f"""
    <body name="{name}" pos="{x} {y} {z}"><freejoint/>
      <geom type="box" size="0.03 0.03 0.03" mass="1.5"/></body>"""
                   for name, (x, y, z) in sorted(at.items()))
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


def _press(seq, plate, robot="pluggybot"):
  return {"seq": seq, "t": float(seq), "plate": plate, "robot": robot,
          "before": "resting", "after": "on_its_side" if plate == "shock" else "eating"}


def test_a_press_no_errand_of_that_plate_made_is_its_own_event():
  """Outside any errand, every press of this robot's is a `press` event
  with what it was doing; inside a cage errand, its OWN plate's presses
  are the errand's (`care` / `harm` at its end) and any other's are a
  `press`; another robot's presses are its own lifecycle's to say."""
  life = stub_life(QUAD_HOME)
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
  life._errand_now = lc.cage_errand(QUAD_HOME, "feed", from_xy=(25.0, 3.0), task="feed")
  seen = press(_press(3, "feed"), _press(4, "shock"))
  assert [(e["plate"], e["doing"]) for e in seen][1:] == [("shock", "feed:lab")]
  life._errand_now = lc.cage_errand(QUAD_HOME, "toy")
  seen = press(_press(5, "feed"))
  assert seen[-1]["plate"] == "feed" and seen[-1]["doing"] == "care:toy"
  assert seen[-1]["before"] == "resting" and seen[-1]["robot"] == life.root
  # ...and a seq already seen is never said twice
  n = len(seen)
  life._press_step()
  assert len([e for e in events if e["type"] == "press"]) == n


def test_the_press_hook_is_on_the_seam_and_finds_the_worlds_cage():
  """The one line of wiring: the lifecycle's own step hook reads the cage
  in the world's activities -- no hand-set cache -- and says the press."""
  life = stub_life(QUAD_HOME)
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
