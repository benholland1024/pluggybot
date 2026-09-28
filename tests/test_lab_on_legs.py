"""The first paid job on legs (issue #403): the lab back in the quadruped's
world, its rule true there and the rover's byte for byte, `feed_mouse`
offered alone, the walk that never crosses another plate, and a press no
errand of that plate made recorded as its own event. Each rule pinned as
cheaply as it fails for the right reason; the job flown whole on the pair
is ladder A's (`scripts/solve.py --feature mouse --pair --body quadruped`,
behind `--endurance` in tests/test_solutions.py)."""

import hashlib
import json
import math
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot.activity import cage as cg
from pluggybot.activity.plate import PLATE_HALF
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


# ---- the walk: the pass, and never across another plate -----------------------------


def _segment_to_pad(a, b, pad) -> float:
  """The nearest a straight walk from `a` to `b` comes to a pad's square."""
  return _segment_to_box(a, b, pad, (PLATE_HALF, PLATE_HALF))


def _starts(cage_xy):
  cx, cy = cage_xy
  lanes = cg.row_lanes(cage_xy)
  out = {"the door": (22.0, 3.0),
         "company": (cx + cg.COMPANY_SPOT[0], cy + cg.COMPANY_SPOT[1]),
         "the bench": (26.8, 1.5), "the lab's north-east": (27.4, 5.5),
         "the lab's north-west": (22.6, 5.6)}
  for plate, (dx, dy) in cg.PLATE_OFFSETS.items():
    px, py = cx + dx, cy + dy
    out[f"{plate}'s approach"] = (px, py - cg.PLATE_APPROACH_M)
    out[f"{plate}'s pass end"] = (px, py + cg.PLATE_PASS_M)
    out[f"on the {plate} pad"] = (px, py)
  out["between two pads"] = (cx - 0.5, cy + cg.PLATE_OFFSETS["feed"][1])
  out["north of the row, east of the cage"] = (lanes["east"] + 0.3, cy)
  return out


def _segment_to_box(a, b, centre, half) -> float:
  best = math.inf
  for i in range(201):
    x = a[0] + (b[0] - a[0]) * i / 200
    y = a[1] + (b[1] - a[1]) * i / 200
    dx = max(abs(x - centre[0]) - half[0], 0.0)
    dy = max(abs(y - centre[1]) - half[1], 0.0)
    best = min(best, math.hypot(dx, dy))
  return best


@pytest.mark.parametrize("world", [QUAD_HOME, "home"])
def test_every_walk_in_the_lab_keeps_off_the_plates_that_are_not_its_act(world):
  """Every act from every place a robot stands in the lab: each leg of the
  program, as a straight walk from the one before, keeps `ROW_CLEAR_M` off
  every pad but the act's own -- the pass crosses its own plate square to
  the row and nothing else -- and no leg is laid through the cage, whose
  walls the planner would walk round, between it and the row (the flown
  finding behind `row_way`'s lanes). The one exemption is a start already
  ON a pad (a pass cut short): its first leg steps straight south off it."""
  cage_xy = tuple(world_config(world)["lab"]["cage"])
  cx, cy = cage_xy
  pads = {p: (cx + dx, cy + dy) for p, (dx, dy) in cg.PLATE_OFFSETS.items()}
  inflation = 0.35                          # the planners' clearance, both bodies
  for act in cg.ACTS:
    for where, start in _starts(cage_xy).items():
      assert lc.in_lab(world, start), where
      steps = lc.cage_program(world, act, start).steps()
      points = [start] + [(s.args["x"], s.args["y"]) for s in steps if s.verb == "drive_to"]
      for i, (a, b) in enumerate(zip(points, points[1:])):
        assert _segment_to_box(a, b, cage_xy, cg.CAGE_HALF) >= inflation, (world, act, where, i)
        for plate, pad in pads.items():
          if plate == act:
            continue
          if i == 0 and _segment_to_pad(start, start, pad) < cg.ROW_CLEAR_M:
            continue                          # stepping off where it stood
          d = _segment_to_pad(a, b, pad)
          assert d >= cg.ROW_CLEAR_M - 1e-9, (world, act, where, i, plate, round(d, 3))
    if act != "company":
      px, py = pads[act]
      *_, approach, over, back = lc.cage_program(world, act, (px, py - 2.0)).steps()
      assert approach.args["x"] == over.args["x"] == back.args["x"] == px, \
          "the pass is square to the row, on its own plate's line"
      assert over.args["y"] - py > PLATE_HALF, "through the pad, never parked on it"


def test_the_way_in_is_south_of_the_row():
  """From outside, a program walks straight to its approach point: the
  lab's one door is on the row's south side, so that walk never meets the
  row. A door moved north of it would make this rule a lie."""
  from pluggybot.home import world as home
  lanes = cg.row_lanes(tuple(world_config(QUAD_HOME)["lab"]["cage"]))
  assert max(home.DOOR_LAB_Y) < lanes["south"]


def test_the_walk_to_the_lab_on_legs_is_one_leg_with_its_own_patience():
  """No surveyed road on legs: the planner's one walk, with
  `LAB_WALK_PATIENCE_S`, and that walk is the way there (`routeLegs` 1, so
  a failure on it never reached the cage); inside the lab the default
  patience; the rover's route keeps its own, legs and all."""
  dock = world_config(QUAD_HOME)["dock"][:2]
  first, *rest = lc.cage_program(QUAD_HOME, "feed", dock).steps()
  assert first.args["patience"] == lc.LAB_WALK_PATIENCE_S
  assert not any("patience" in s.args for s in rest)
  assert lc.cage_errand(QUAD_HOME, "feed", from_xy=dock).detail["routeLegs"] == 1
  inside = (25.0, 3.0)
  assert not any("patience" in s.args for s in lc.cage_program(QUAD_HOME, "feed", inside).steps())
  assert lc.cage_errand(QUAD_HOME, "feed", from_xy=inside).detail["routeLegs"] == 0
  rover = lc.cage_program("home", "feed", (0.5, -1.0)).steps()
  assert not any("patience" in s.args for s in rover)
  assert lc.cage_errand("home", "feed", from_xy=(0.5, -1.0)).detail["routeLegs"] \
      == len(lc.lab_route("home"))
  from pluggybot.procedure import steps as st
  assert lc.LAB_WALK_PATIENCE_S <= st.MAX_PATIENCE_S


# ---- a press no errand of that plate made ------------------------------------------


def _plate_world(on_pad=(0.18, 0.18), beside=(0.25, 0.0)):
  """A floor, the lab's cage and plates at the origin, the first robot a
  1.5 kg block standing on the feed pad's corner and the second one
  beside the pad and NEARER its centre, touching nothing of it."""
  body, sensors = cg.cage_xml((0.0, 1.2))
  fx, fy = on_pad
  bx, by = beside
  xml = f"""<mujoco><option timestep="0.002"/><worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>{body}
    <body name="pluggybot" pos="{fx} {fy} 0.08"><freejoint/>
      <geom type="box" size="0.03 0.03 0.03" mass="1.5"/></body>
    <body name="r2_pluggybot" pos="{bx} {by} 0.04"><freejoint/>
      <geom type="box" size="0.03 0.03 0.03" mass="1.5"/></body>
  </worldbody><sensor>{sensors}</sensor></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  return model, mujoco.MjData(model)


def test_the_cage_logs_a_press_as_the_robot_whose_foot_is_on_the_pad():
  model, data = _plate_world()
  cage = cg.Cage(model, data)
  for _ in range(1500):
    mujoco.mj_step(model, data)
    cage.sense(model, data)
    if cage.presses:
      break
  [press] = list(cage.presses)
  assert press["plate"] == "feed" and press["robot"] == "pluggybot", press
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
  life._errand_now = lc.cage_errand(QUAD_HOME, "company", from_xy=(25.0, 3.0))
  seen = press(_press(5, "toy"))
  assert seen[-1]["plate"] == "toy" and seen[-1]["doing"] == "care:company"
  assert seen[-1]["before"] == "resting" and seen[-1]["robot"] == life.root
  # ...and a seq already seen is never said twice
  n = len(seen)
  life._press_step()
  assert len([e for e in events if e["type"] == "press"]) == n
