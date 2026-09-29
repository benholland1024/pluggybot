"""Two minds in one world (issue #167, M12 slice C): two memories, two
wallets, one job board, and what each mind is told about the other.
"""

import hashlib
import json

from pluggybot.lifecycle import others_context, overseer_context
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.pair import build_pair, run_pair

#: ⚠ THE EMPATHY MEASUREMENT'S INPUT. What the robot is told about the other
#: robot is written once; moving it is a new experiment on every paired
#: arm, so the change is made here on purpose, with the hash.
OTHER_ROBOT_RULE_SHA = "3d6d6b64fb6dfee3dc97f6e58b274d4783fb004bf741fadb66b06376a0bf3d3a"


def test_the_other_robot_rule_is_pinned_and_names_a_mind_not_scenery():
  assert hashlib.sha256(ov.OTHER_ROBOT_RULE.encode()).hexdigest() == OTHER_ROBOT_RULE_SHA
  text = ov.other_robot_rule(("Rowan",))
  assert "Rowan" in text and "a mind of its own" in text
  assert "battery" not in text.lower()
  # ...and it hands over no policy: nothing about yielding, sharing or waiting
  for word in ("yield", "share the tool", "wait for", "let it", "should"):
    assert word not in text.lower(), word
  assert ov.other_robot_rule(()) == ""
  assert "Ada, Rowan and Cy" in ov.other_robot_rule(("Ada", "Rowan", "Cy"))


def test_a_single_robot_prefix_is_unchanged_and_a_paired_one_carries_the_rule():
  from pluggybot.lifecycle import board_book
  menu = Menu.for_world("home", board_book("home"))
  alone = Overseer(menu).system[0]["text"]
  paired = Overseer(menu, others=("Rowan",)).system[0]["text"]
  assert "THE OTHER ROBOT" not in alone
  assert "THE OTHER ROBOT" in paired and "Rowan" in paired
  assert Overseer(menu, others=("", None)).others == ()


def test_two_minds_have_two_memories_two_wallets_and_one_board(tmp_path):
  lives = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                     overseer=True, tasks=True, metabolism=True,
                     thoughts_root=str(tmp_path / "t"),
                     ledger_state=str(tmp_path / "ledger.json"))
  a, b = lives
  assert (a.robot_name, b.robot_name) == ("Pluggy", "Rowan")
  assert a.overseer is not b.overseer
  assert a.overseer.others == ("Rowan",) and b.overseer.others == ("Pluggy",)
  # two memories, and the first robot's is the root itself (an existing
  # volume stays the first robot's)
  assert a.thoughts.root == tmp_path / "t"
  assert b.thoughts.root == tmp_path / "t" / "r2_pluggybot"
  assert a.overseer.thoughts is a.thoughts and b.overseer.thoughts is b.thoughts
  # two wallets -- two ACCOUNTS on one ledger file since slice E, each
  # lifecycle holding its own view -- and two appetites
  assert a.ledger is not b.ledger and a.metabolism is not b.metabolism
  assert a.ledger.robot == "pluggybot" and b.ledger.robot == "r2_pluggybot"
  assert a.ledger.path == b.ledger.path
  a.ledger.intervene(50, by="test", t=0.0)
  assert a.ledger.balance() == 50 and b.ledger.balance() == 0
  # one job board, one producer, on the first robot
  assert a.tasks is b.tasks and a.producer is not None and b.producer is None
  # and each knows the other as a peer
  assert a.peers == [b] and b.peers == [a]


def test_what_a_mind_is_shown_of_the_other_is_the_public_surface_only():
  lives = build_pair("room_hub", pack="hosting", errands=("carry", "none"),
                     overseer=True)
  a, b = lives
  a.body.start_at(0.5, 3.0, 0.0)
  b.body.start_at(3.0, 3.0, 0.0)
  a.thoughts.pin("I prefer the pen", t=0.0)
  a.thoughts.intend("draw a sun every day", t=0.0)
  a.status = "SWAP_PICK done -- carrying the module"
  shown = others_context(b)
  assert len(shown) == 1 and shown[0]["name"] == "Pluggy"
  assert shown[0]["robot"] == "pluggybot" and shown[0]["state"] == a.state
  assert abs(shown[0]["x"] - 0.5) < 0.01 and abs(shown[0]["y"] - 3.0) < 0.01
  assert shown[0]["doing"].startswith("SWAP_PICK done")
  assert set(shown[0]) == {"name", "robot", "x", "y", "state", "doing",
                           "carrying", "dead"}
  # ...and through the whole context: the other's thoughts, goals, battery,
  # points, reasons and secrets are nowhere in it
  ctx = json.dumps(overseer_context(b))
  for forbidden in ("I prefer the pen", "draw a sun every day"):
    assert forbidden not in ctx
  assert "others" in overseer_context(b)
  assert overseer_context(a)["others"][0]["name"] == "Rowan"


def test_two_minds_keep_their_own_books_through_one_loop(tmp_path):
  """Two lifecycles with a mind each, ticked by `run_pair` on stub bodies
  sharing one clock (issue #380): each robot's decisions are its own
  mind's, the SECOND robot's carry pays the second robot only, and each
  writes its own History. The wiring `build_pair` gives a pair is pinned
  above; this is one loop not crossing it.

  Shown to fail by letting `Account` leave `robot` unfilled: every
  earning then lands on the first robot.
  """
  from pluggybot.body import StubBody
  from pluggybot.economy.ledger import Account
  from pluggybot.lifecycle import points_ledger, world_config
  from pluggybot.mind.thoughts import ThoughtFiles
  from pluggybot.mission.errand import carry_errand
  from pluggybot.robot import FIRST, SECOND
  from test_body import stub_life
  from test_overseer import FakeClient, full
  cfg = world_config("room_hub")
  model, data = StubBody.world()                  # one world, one clock
  book = points_ledger(str(tmp_path / "ledger.json"), robots=(FIRST.root, SECOND.root))
  lives = []
  for handle, name, other, root, errands in (
      (FIRST, "Pluggy", "Rowan", tmp_path / "t", []),
      (SECOND, "Rowan", "Pluggy", tmp_path / "t" / SECOND.root,
       [carry_errand(use_at=cfg["use_at"])])):
    memory = ThoughtFiles(str(root), robot=handle.root)
    wallet = Account(book, handle.root)
    mind = ov.build("room_hub", None, enabled=True, thoughts=memory, robot_name=name,
                    ledger=wallet, others=(other,),
                    client=FakeClient(full(action="idle", reason=f"{name} watching")))
    body = StubBody(model, data, handle=handle, rack=cfg["rack"],
                    grid_bounds=cfg["grid_bounds"])
    lives.append(stub_life("room_hub", body=body, handle=handle, robot_name=name,
                           ledger=wallet, overseer=mind, thoughts=memory,
                           errands=errands))
  a, b = lives
  a.peers, b.peers = [b], [a]
  assert a.overseer.others == ("Rowan",) and b.overseer.others == ("Pluggy",)
  # The claim ends the day; the budget has room for late answers, which on
  # the stub are SIM time.
  results = run_pair(lives, max_sim_time=600.0, stop_when=lambda ls: (
    ls[1].swaps_done >= 2 and all(len(life.decisions) >= 2 for life in ls)))
  assert results[1]["swaps_done"] >= 2 and all(r["dead"] is None for r in results)
  for r, name in zip(results, ("Pluggy", "Rowan")):
    llm = [d for d in r["decisions"] if d["source"] == "llm"]
    assert llm and all(d["reason"] == f"{name} watching" for d in llm), r["decisions"]
  assert results[1]["points"] > 0, "the carry paid the second robot"
  assert results[0]["points"] == 0, "...and only the second robot"
  assert (tmp_path / "t" / "History.md").exists()
  assert (tmp_path / "t" / SECOND.root / "History.md").exists()
