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
OTHER_ROBOT_RULE_SHA = "6e2215bec757f10a9f402c3e24ef2d0856b6466fcdbb596595807040fffea62b"

WORLD = "home_quad"


def stub_pair(root=None, ledger_state=None, *, tasks=False,
              metabolism=False, inboxes=None, names=("Pluggy", "Rowan"),
              clients=None, errands=((), ())):
  """Two lifecycles on stub bodies in one world, wired as `build_pair`
  wires a pair's BOOKKEEPING: one ledger with an account each, a memory
  each (the second under the first's root), a mind each told the other's
  name, one board whose producer is the first robot's, one rack inventory,
  and each the other's peer. The bodies' mutual awareness and the world's
  activities are left out: nothing here moves. That the two wirings agree
  is `test_two_minds_have_two_memories_two_wallets_and_one_board`'s."""
  from pathlib import Path

  from pluggybot.body import StubBody
  from pluggybot.economy import energy
  from pluggybot.economy.cadence import default_cadence
  from pluggybot.economy.ledger import Account
  from pluggybot.economy.metabolism import Appetite, Metabolism
  from pluggybot.lifecycle import (
    board_book, points_ledger, task_board, task_producer, world_config,
  )
  from pluggybot.mind.thoughts import ThoughtFiles
  from pluggybot.robot import FIRST, SECOND
  from test_body import stub_life
  from test_overseer import FakeClient
  cfg = world_config(WORLD)
  model, data = StubBody.world()                  # one world, one clock
  book = board_book(WORLD)
  appetite = Appetite.load(WORLD) if metabolism else None
  ledger = points_ledger(ledger_state, cap=appetite.cap if appetite else None,
                         robots=(FIRST.root, SECOND.root))
  beat = default_cadence(WORLD) if tasks else None
  board = task_board(cadence=beat, world=WORLD) if tasks else None
  maker = (task_producer(board, WORLD, book, beat, procedures=True, robots=names)
           if tasks else None)
  lives = []
  for i, (handle, name) in enumerate(zip((FIRST, SECOND), names)):
    where = (None if root is None else
             str(root) if i == 0 else str(Path(root) / handle.root))
    memory = ThoughtFiles(where, robot=handle.root)
    wallet = Account(ledger, handle.root)
    hunger = Metabolism(ledger, appetite, robot=handle.root) if appetite else None
    mind = ov.build(WORLD, book, enabled=True, thoughts=memory, robot_name=name,
                    ledger=wallet, appetite=hunger is not None,
                    others=tuple(n for n in names if n != name),
                    client=(clients or (FakeClient(), FakeClient()))[i])
    body = StubBody(model, data, handle=handle, rack=cfg["rack"],
                    grid_bounds=cfg["grid_bounds"], charge_w=energy.load(WORLD).charge_w)
    lives.append(stub_life(WORLD, body=body, handle=handle, robot_name=name,
                           ledger=wallet, overseer=mind, thoughts=memory,
                           metabolism=hunger, tasks=board, boards=book,
                           producer=maker if i == 0 else None,
                           inbox=inboxes[i] if inboxes else None,
                           errands=list(errands[i])))
  a, b = lives
  a.expects_work = b.expects_work = maker is not None
  b.rack_inventory, b.lost_tool_after_s = a.rack_inventory, None
  a.peers, b.peers = [b], [a]
  return a, b


def wiring(lives) -> dict:
  """What a pair's books share and what each robot keeps, as facts."""
  a, b = lives
  return {
    "names": (a.robot_name, b.robot_name),
    "others": (a.overseer.others, b.overseer.others),
    "minds": a.overseer is not b.overseer,
    "memories": (a.overseer.thoughts is a.thoughts, b.overseer.thoughts is b.thoughts,
                 b.thoughts.root.relative_to(a.thoughts.root).as_posix()),
    "wallets": (a.ledger.robot, b.ledger.robot, a.ledger.path == b.ledger.path,
                a.ledger is not b.ledger),
    "appetites": a.metabolism is not b.metabolism and None not in (a.metabolism,
                                                                    b.metabolism),
    "board": (a.tasks is b.tasks, a.producer is not None, b.producer is None),
    "targets": a.producer.targets.get("robot"),
    "rack": a.rack_inventory is b.rack_inventory,
    "work": (a.expects_work, b.expects_work, b.lost_tool_after_s),
    "peers": (a.peers == [b], b.peers == [a]),
    "arm": (a.autonomous, b.autonomous, a.overseer._acts(), b.overseer._acts()),
  }


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
  menu = Menu.for_world(WORLD, board_book(WORLD))
  alone = Overseer(menu).system[0]["text"]
  paired = Overseer(menu, others=("Rowan",)).system[0]["text"]
  assert "THE OTHER ROBOT" not in alone
  assert "THE OTHER ROBOT" in paired and "Rowan" in paired
  assert Overseer(menu, others=("", None)).others == ()


def test_two_minds_have_two_memories_two_wallets_and_one_board(tmp_path):
  """`build_pair`'s wiring, on the served pair -- and the stub pair every
  other paired test here is built on is wired the same way."""
  lives = build_pair(WORLD, pack="hosting", errands=("none", "none"),
                     overseer=True, tasks=True, metabolism=True,
                     names=("Pluggy", "Rowan"),
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
  # one job board, one producer, on the first robot, naming both robots
  assert a.tasks is b.tasks and a.producer is not None and b.producer is None
  assert a.producer.targets["robot"] == ["Pluggy", "Rowan"]
  # and each knows the other as a peer
  assert a.peers == [b] and b.peers == [a]
  stub = stub_pair(tmp_path / "s", str(tmp_path / "s.json"),
                   tasks=True, metabolism=True)
  assert wiring(stub) == wiring(lives)


def test_what_a_mind_is_shown_of_the_other_is_the_public_surface_only():
  a, b = stub_pair()
  a.body.start_at(0.5, 3.0, 0.0)
  b.body.start_at(3.0, 3.0, 0.0)
  a.thoughts.pin("I prefer the pen", t=0.0)
  a.thoughts.intend("draw a sun every day", t=0.0)
  a.status = "EXPLORE: heading for the kitchen"
  shown = others_context(b)
  assert len(shown) == 1 and shown[0]["name"] == "Pluggy"
  assert shown[0]["robot"] == "pluggybot" and shown[0]["state"] == a.state
  assert abs(shown[0]["x"] - 0.5) < 0.01 and abs(shown[0]["y"] - 3.0) < 0.01
  assert shown[0]["doing"].startswith("EXPLORE: heading")
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
  mind's, the SECOND robot's paid job pays the second robot only, and each
  writes its own History. The wiring is pinned above; this is one loop not
  crossing it.

  Shown to fail by letting `Account` leave `robot` unfilled: every
  earning then lands on the first robot.
  """
  from pluggybot.mission.errand import Errand
  from pluggybot.robot import SECOND
  from test_overseer import FakeClient, full
  carry = Errand(name="carry:module_lcd", module="module_lcd", station_y=0.0,
                 use_at=(1.0, 1.0), needs_use_pose=False)
  lives = stub_pair(tmp_path / "t", str(tmp_path / "ledger.json"),
                    clients=tuple(FakeClient(full(action="idle", reason=f"{name} watching"))
                                  for name in ("Pluggy", "Rowan")),
                    errands=((), (carry,)))
  a, b = lives
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
