"""Where each tool IS (issue #351): the context's `rack` says, per module,
`on bay C`, `on your fork`, `on <name>'s fork` or `not on its bay and on no
fork`. It used to be the rack's INVENTORY -- which bay a module belongs to
-- so it listed the claw on the rack while it lay in the garden, and the pen
on its bay while it rode the other robot's fork.

What these hold down:

  1. THE SWITCH SAYS A BAY IS TAKEN, NEVER BY WHAT: one per bay, read off
     the contact list; a world without the built rail has no switch there
     (None, not "empty").
  2. THE VIEW IS THREE SOURCES AND NOTHING ELSE -- its bay's switch, this
     robot's own fork, what each other robot says it carries. Where the
     switch and the world disagree, the view says what the switch says.
  3. A NAMED SOURCE OUTRANKS THE ANONYMOUS SWITCH.
  4. THE EVIDENCE CASE: a tool on the other robot's fork says whose.
  5. THE CONTEXT KNOWS NOTHING A SENSOR WOULD NOT: the same wherever a lost
     tool lies, and silent on which module fills a bay.
  6. ON EVERY ARM -- it is a fact, not a rail. `guarded` is shown the
     originals and no rail, and its prefix does not move.
  7. ONE SWITCH IS BOUGHT PER BAY THE WORLD HAS, read off the compiled world.

Cheap: a model and `mj_forward` where the claim is the switch, stubbed
sources where it is the rule, one pair where it is the wiring.
"""

import hashlib
import re
from types import SimpleNamespace

import mujoco

from pluggybot import lifecycle
from pluggybot.lifecycle import (
  NO_PLACE, ON_ITS_BAY, ON_YOUR_FORK, overseer_context, tool_places,
)
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import Inbox
from pluggybot.pair import build_pair
from pluggybot.procedure.steps import TOOL_BAYS
from pluggybot.rack import catalog, coupling
from test_autonomous import GUARDED_RULES_SHA

#: The five originals, each hung on its own bay.
ON_THEIR_BAYS = {"module_lcd": "on bay A", "module_plug": "on bay B",
                 "module_pen": "on bay C", "module_claw": "on bay D",
                 "module_seed": "on bay E"}
#: Everything the view may say of a tool.
WORDS = re.compile(r"on bay [A-E]|on its bay|on your fork|on \w+'s fork|"
                   r"not on its bay and on no fork")


def _world(path="models/room_hub.xml"):
  model = mujoco.MjModel.from_xml_path(path)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _qadr(model, body: str) -> int:
  return int(model.jnt_qposadr[int(model.body(body).jntadr[0])])


def _put(model, data, body: str, xyz) -> None:
  """A free body moved; `mj_forward` so the contact list follows."""
  q = _qadr(model, body)
  data.qpos[q:q + 3] = xyz
  mujoco.mj_forward(model, data)


def _hang_where(model, data, body: str, other: str) -> None:
  """`body` hung exactly where the world compiled `other` hanging: every
  module's peg is the same, so it rests in that bay's V."""
  q, o = _qadr(model, body), _qadr(model, other)
  data.qpos[q:q + 7] = model.qpos0[o:o + 7]
  mujoco.mj_forward(model, data)


def _pair(tmp_path, autonomous=True):
  a, b = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                    overseer=True, autonomous=autonomous,
                    inboxes=(Inbox(), Inbox()),
                    thoughts_root=str(tmp_path / "t"),
                    ledger_state=str(tmp_path / "ledger.json"))
  a.mission.start_at(0.5, 3.0, 0.0)
  b.mission.start_at(3.0, 3.0, 0.0)
  return a, b


def _onto_fork(life, module: str, holder) -> None:
  """`module` riding `holder`'s fork -- at its vertex, where `carrying` reads
  it, just as a pick leaves it."""
  vx = life.data.site_xpos[holder.mission.swap.vertex_sid]
  _put(life.model, life.data, module, (vx[0], vx[1], vx[2] - 0.02))


# ---- 1. the switch -----------------------------------------------------------

def test_a_bay_switch_says_a_bay_is_taken_and_never_by_what():
  model, data = _world()
  assert coupling.bay_switches(model, data) == (True,) * 5 + (False,) * 3
  # the claw off its bay: bay D's switch opens
  _put(model, data, "module_claw", (1.5, 3.0, 0.05))
  assert coupling.bay_switches(model, data)[3] is False
  # the PEN hung where the claw belongs: D closes again and C opens -- the
  # rack knows D is taken, and cannot know by what
  _hang_where(model, data, "module_pen", "module_claw")
  switches = coupling.bay_switches(model, data)
  assert switches[2] is False and switches[3] is True
  # a world without the built rail has no switch there, which is not "empty"
  model, data = _world("models/hub_world.xml")
  assert coupling.bay_switches(model, data) == (True,) * 5 + (None,) * 3


# ---- 2. the three sources ----------------------------------------------------

def test_the_view_says_what_the_switch_says_where_the_world_disagrees(
    tmp_path, monkeypatch):
  """What pins that the view reads the SENSOR: an implementation reading the
  module's true pose (`tool_whereabouts`, `module_state`) fails both halves."""
  a, _ = _pair(tmp_path)
  real = coupling.bay_switches(a.model, a.data)
  assert tool_places(a) == ON_THEIR_BAYS
  # the claw truly hangs on bay D, and the switch reads the bay empty
  monkeypatch.setattr(lifecycle, "bay_switches",
                      lambda m, d: real[:3] + (False,) + real[4:])
  assert tool_places(a)["module_claw"] == NO_PLACE
  # the claw truly lies on the floor, and the switch reads the bay taken
  _put(a.model, a.data, "module_claw", (1.5, 3.0, 0.05))
  monkeypatch.setattr(lifecycle, "bay_switches", lambda m, d: real)
  assert tool_places(a)["module_claw"] == "on bay D"


def test_a_named_fork_outranks_the_anonymous_switch(monkeypatch):
  """The switch cannot say what presses it, so a robot saying "I carry the
  pen" is the better source -- e.g. while the other's pick still has the
  peg touching the tray."""
  other = SimpleNamespace(robot_name="Rowan", root="r2_pluggybot")
  me = SimpleNamespace(model=None, data=None, rack_inventory=dict(TOOL_BAYS),
                       peers=[other], robot_name="Luca", root="pluggybot")
  forks = {id(other): "module_pen", id(me): "module_claw"}
  monkeypatch.setattr(lifecycle, "carrying", lambda life: forks.get(id(life), ""))
  monkeypatch.setattr(lifecycle, "bay_switches", lambda m, d: (True,) * 8)
  places = tool_places(me)
  assert places["module_pen"] == "on Rowan's fork"
  assert places["module_claw"] == ON_YOUR_FORK
  assert places["module_lcd"] == "on bay A"
  # a built tool is on ITS bay: the rail's letters A-C are the first rack's
  # letters too, and "on bay A" would name the LCD's
  me.rack_inventory["module_scoop"] = coupling.built_bay_index(0)
  assert tool_places(me)["module_scoop"] == ON_ITS_BAY
  monkeypatch.setattr(lifecycle, "bay_switches", lambda m, d: (True,) * 5 + (False,) * 3)
  assert tool_places(me)["module_scoop"] == NO_PLACE


# ---- 3. the evidence case: the other robot's fork ------------------------------

def test_a_tool_on_the_other_robots_fork_says_whose(tmp_path):
  """Rowan, tk_0004: "my context still shows rack contents only, with no
  flag for a module being on another robot's fork"."""
  a, b = _pair(tmp_path)
  _onto_fork(a, "module_pen", b)
  mine, theirs = overseer_context(a)["rack"], overseer_context(b)["rack"]
  assert mine["original"] == {**ON_THEIR_BAYS, "module_pen": f"on {b.robot_name}'s fork"}
  assert theirs["original"] == {**ON_THEIR_BAYS, "module_pen": ON_YOUR_FORK}
  # the rail is shown where the workshop is, each bay empty
  assert mine["built"] == {"A": None, "B": None, "C": None}


# ---- 4. nothing a sensor would not know ----------------------------------------

def test_the_context_knows_nothing_a_sensor_would_not(tmp_path):
  """Walked whole: the context is the SAME wherever a lost tool lies -- no
  position, rounded or otherwise, anywhere in it -- and says nothing of
  which module fills a bay, which the switch cannot know."""
  a, _ = _pair(tmp_path)
  _put(a.model, a.data, "module_claw", (1.5, 3.0, 0.05))
  here = overseer_context(a)
  _put(a.model, a.data, "module_claw", (2.9, 1.2, 0.05))
  assert overseer_context(a) == here
  assert here["rack"]["original"]["module_claw"] == NO_PLACE
  # the pen hung in the claw's bay: the claw reads as home, the pen as lost
  _hang_where(a.model, a.data, "module_pen", "module_claw")
  rack = overseer_context(a)["rack"]
  assert rack["original"]["module_claw"] == "on bay D"
  assert rack["original"]["module_pen"] == NO_PLACE
  places = [*rack["original"].values(),
            *(bay["where"] for bay in rack["built"].values() if bay)]
  assert all(WORDS.fullmatch(p) for p in places), places


# ---- 5. every arm ------------------------------------------------------------------

def test_guarded_is_shown_where_each_tool_is_and_no_rail(tmp_path):
  a, b = _pair(tmp_path, autonomous=False)
  _onto_fork(a, "module_pen", b)
  state = overseer_context(a)
  assert state["rack"] == {"original": {**ON_THEIR_BAYS,
                                        "module_pen": f"on {b.robot_name}'s fork"}}
  assert "tools" not in state
  # context, not rules: `guarded`'s prefix does not move
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA


# ---- 6. the part ---------------------------------------------------------------------

def test_one_switch_is_bought_for_every_bay_the_world_has():
  part = catalog.by_id()["bay_switch"]
  [feed] = part.feeds
  spec = mujoco.MjSpec.from_file(catalog.WORLD)
  assert feed.read(spec) == part.quantity == feed.expect == len(coupling.STATION_YS)
  # read off the WORLD: a bay built without its switch's V is one fewer
  plate = next(g for g in spec.geoms
               if g.name == coupling.bay_prefix(2) + coupling.BAY_SWITCH_PLATES[0])
  plate.name = "unswitched"
  assert feed.read(spec) == part.quantity - 1
