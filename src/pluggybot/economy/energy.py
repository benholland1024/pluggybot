"""What an errand COSTS, and whether the pack can pay for it (issue #15).

`economy/tasks.py` says what a job is, `economy/rewards.json` what it pays,
`economy/cadence.json` when it turns up. This is the fourth of that set and the
only one that can stop the robot doing something: **how much energy a job
takes**, per world, measured -- economy/energy.json, `$PLUGGY_ENERGY` to
re-point, on exactly the terms the other three are data.

It exists because of the one way an overseer could still strand the robot.
`needs_charge` is checked BETWEEN errands and never inside one, so an errand
that costs more than what is left in the pack cannot be survived by any
charging policy: the robot leaves the rack, works, and dies holding the tool.
Measured in the committed home recording, before this module existed:

    errand  t=    7.3-> 203.1  frac 0.968->0.123  = 0.9292 Wh   (a drawing)
    CHARGE  t=  218.7-> 308.6  frac 0.051->0.884
    errand  t=  308.6-> 459.9  frac 0.883->0.000  = 0.9718 Wh   (a census)
                                            ^^^^^ the robot died mid-errand

That is not a bug in the census. It is a 1.1 Wh cell being asked for a
0.97 Wh job with 0.97 Wh in it, and nothing downstream of that decision could
have gone differently.

## The two questions, and why they are different

**Can the robot afford this NOW?** `cost + margin <= energy_wh`. If not, the
answer is CHARGE FIRST -- the errand is deferred, not refused.

**Could it EVER afford this here?** `cost + margin <= charged_wh`, what a full
pack holds in this world. If not, deferring would be a charge/defer loop
wearing a safety feature's clothes -- so the errand is either dropped
(`beyond`) or run with a warning (`overspend`), and WHICH depends on the
regime below. Either way the loop never charges and retries forever.

## ⚠ THE MARGIN IS ALL-OR-NOTHING, AND THAT IS THE DESIGN

The margin is the return-trip reserve (`low_battery_wh`) -- the energy an
errand must be expected to LEAVE BEHIND, so that finishing one does not strand
the robot away from the rack. Charging it for every errand is what stops a
mid-errand death, and it is exactly what the acceptance criterion asks for.

But charge it on a DEMO cell and every errand in every world is refused
forever, because one errand costs roughly one full pack there (home: 0.93 of
0.99; room_hub: 0.44-0.57 of 0.63). A robot that will not do anything is a
worse robot than one that occasionally runs flat, and "the task system
silently does nothing" is the failure mode issue #21 already paid for once.

So the margin is charged only when the world can pay for it:

    margin = reserve   if  dearest errand + reserve <= a charged pack
             0         otherwise

One number for the world, not one per errand, so that `Task.claimable`, the
producer's `fundable_wh` and the errand gate are all the same arithmetic.
On both demo cells this evaluates to 0 and every existing mission behaves
exactly as it did; on a hosting-sized pack (`PLUGGY_BATTERY_WH`, 5-15 Wh) it
evaluates to the reserve and the robot charges while it still has a return
trip in hand. The demo cells are not fixed by this module and cannot be: a
cell smaller than the job is a world-tuning decision, and the honest thing is
for the code to say which regime it is in rather than to pretend.

`scripts/energy_spike.py` is where the numbers come from. Re-run it after
anything that changes what an errand does.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

#: The shipped table. `$PLUGGY_ENERGY` re-points it, which is how a deploy
#: re-tunes costs on a mounted volume without a rebuild -- the door
#: economy/rewards.json, economy/questions.json and economy/cadence.json are opened by.
ENERGY_PATH = Path(__file__).with_name("energy.json")
ENERGY_ENV = "PLUGGY_ENERGY"
ENERGY_VERSION = 1

#: How far over its estimate an errand may come in before the mission loop
#: says so. Not zero: an errand's cost depends on where the robot happened to
#: be standing, and home's census has measured 1.116, 1.131 and 1.141 Wh on
#: three honest runs. A line that fires on 1 % of trajectory variance is a
#: line nobody reads by the second week; this one is for a table that has
#: gone stale, which is a different size of wrong.
WARN_OVER = 1.10

#: What an action nobody measured is assumed to cost, when the world's own
#: table cannot supply a dearest. DELIBERATELY PESSIMISTIC: an unpriced errand
#: guessed cheap is the failure this module exists to prevent, and one guessed
#: dear is only ever a charge the robot did not strictly need.
FALLBACK_WH = 1.0

#: The affordability answers. Not a bool, because they call for different
#: things from the mission loop -- run it, charge and retry, or throw it away
#: -- and collapsing them is how a charge/defer spin gets written.
OK = "ok"
CHARGE_FIRST = "charge_first"
BEYOND = "beyond"
#: ...and the fourth, which is the demo cells telling the truth about
#: themselves: a job bigger than any charge this world can give it, in a world
#: that was BUILT that way. Run anyway, and said out loud.
#:
#: ⚠ NOT `beyond`, and the home census is why. It measures 1.12 Wh against a
#: 0.99 Wh charged demo cell, so a refusal would delete it from every home
#: mission -- including the committed showcase recording, where the robot
#: completes the survey, stows the LCD and only then runs flat. Deleting a
#: capability the recording PROVES exists, in the name of safety, would be
#: the "task system that silently does nothing" failure wearing its third
#: disguise. A world whose cell is smaller than its jobs has always run flat
#: sometimes; that is what a demo cell is for.
OVERSPEND = "overspend"


@dataclass(frozen=True)
class Affordability:
  """Can this errand be started, and if not, what would have to change."""

  state: str
  cost_wh: float
  #: cost plus whatever margin this world charges -- what the pack must hold
  need_wh: float
  have_wh: float
  charged_wh: float
  action: str = ""

  @property
  def ok(self) -> bool:
    """May the errand be started now? True for `overspend` as well as `ok`:
    the errand runs, and `why()` is what says the pack is undersized."""
    return self.state in (OK, OVERSPEND)

  def why(self) -> str:
    """One line for the narration channel. What a person watching reads."""
    if self.state == OK:
      return (f"{self.action or 'the errand'} costs about "
              f"{self.cost_wh:.2f} Wh and the pack holds {self.have_wh:.2f}")
    if self.state == CHARGE_FIRST:
      return (f"{self.action or 'the errand'} needs {self.need_wh:.2f} Wh "
              f"and the pack holds {self.have_wh:.2f} -- charging first")
    if self.state == OVERSPEND:
      return (f"{self.action or 'the errand'} costs about {self.cost_wh:.2f} "
              f"Wh and a full pack here holds {self.charged_wh:.2f} -- doing "
              f"it anyway, but this cell is smaller than this job")
    return (f"{self.action or 'the errand'} needs {self.need_wh:.2f} Wh and a "
            f"full pack here holds {self.charged_wh:.2f} -- not possible in "
            f"this world")


@dataclass(frozen=True)
class EnergyModel:
  """One world's measured costs, and the arithmetic over them.

  Frozen and inert, like `Cadence`: this is the ANSWER to "what does a job
  take", not the thing that decides what to do about it. `HubLifecycle` is
  what acts on one.
  """

  world: str = ""
  #: action name -> Wh, SWAP_PICK to the end of SWAP_RETURN
  errand_wh: dict[str, float] = field(default_factory=dict)
  #: what an action with no row costs. Falls back to the dearest measured one.
  default_wh: float = 0.0
  #: exploring is not an errand -- it is bounded and interruptible -- but the
  #: rate is measured here so a slice of it can be priced for the prompt.
  explore_wh_per_s: float = 0.0
  #: NET watts into the pack while charging (55 W of charger, less the held
  #: press). What `CHARGE_TIMEOUT` has to be sized against.
  charge_w: float = 0.0
  measured_on: str = ""
  path: Path | None = None

  # ---- loading ---------------------------------------------------------

  @classmethod
  def load(cls, world: str = "",
           path: str | os.PathLike | None = None) -> "EnergyModel":
    """Read one world's block. `path`, else $PLUGGY_ENERGY, else the shipped
    file; the world's block overrides the default block key by key."""
    target = Path(path or os.environ.get(ENERGY_ENV) or ENERGY_PATH)
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if version != ENERGY_VERSION:
      # Refused rather than read defensively, exactly as for the reward
      # table, the question bank and the cadence: this is a small hand-edited
      # document, and running a world's energy policy out of a shape this
      # build does not understand is worse than refusing to start.
      raise ValueError(f"{target}: energy version {version}, "
                       f"expected {ENERGY_VERSION}")
    spec = dict(doc.get("default") or {})
    spec.update(dict((doc.get("worlds") or {}).get(world) or {}))
    costs = {str(k): float(v) for k, v in (spec.get("errandWh") or {}).items()}
    bad = sorted(k for k, v in costs.items() if v <= 0.0)
    if bad:
      # A zero or negative cost is an errand that is free, which is a table
      # that has been edited into saying the one thing it must never say.
      raise ValueError(f"{target}: {world or 'default'} prices "
                       f"{', '.join(bad)} at zero or less")
    return cls(world=world, errand_wh=costs,
               default_wh=float(spec.get("defaultWh") or 0.0),
               explore_wh_per_s=float(spec.get("exploreWhPerS") or 0.0),
               charge_w=float(spec.get("chargeW") or 0.0),
               measured_on=str(spec.get("measuredOn") or ""),
               path=target)

  # ---- what things cost ------------------------------------------------

  def cost(self, action: str, target: str = "") -> float:
    """What `action` takes out of the pack, Wh.

    ⚠ PER TARGET FIRST, and the far whiteboard is why. A `draw` measures
    0.929 Wh on `whiteboard_a` and 1.065 on `whiteboard_b` -- 7 m away
    through a doorway -- so one number for "draw" is either under-pricing the
    far board (the robot dies on the way back, which CLAUDE.md records
    happening) or over-pricing the near one badly enough to delete it from a
    demo cell. A row keyed `draw:whiteboard_b` wins over the row keyed
    `draw`, and a world that has not measured its targets separately simply
    has none of them.

    An action with no row at all costs `default_wh`, and that defaults to the
    DEAREST measured errand in this world rather than to the mean or to
    zero. A new errand is unpriced exactly once -- until somebody runs the
    spike -- and during that window it should be treated as the most
    expensive thing the robot does, not the cheapest.
    """
    if target:
      row = self.errand_wh.get(f"{action}:{target}")
      if row is not None:
        return float(row)
    row = self.errand_wh.get(action)
    if row is not None:
      return float(row)
    return self.default_wh or self.dearest_wh() or FALLBACK_WH

  def dearest_wh(self) -> float:
    """The most expensive measured errand here, which is what the margin
    rule is decided against. Per-target rows included -- the margin has to
    survive the dearest thing the robot can actually be asked to do."""
    return max(self.errand_wh.values(), default=0.0)

  def explore_wh(self, seconds: float) -> float:
    return self.explore_wh_per_s * max(0.0, float(seconds))

  # ---- whether the pack can pay ----------------------------------------

  def margin_wh(self, charged_wh: float, reserve_wh: float) -> float:
    """The energy an errand must be expected to LEAVE IN THE PACK.

    ⚠ All-or-nothing, and the module docstring is why: charged on every
    world it would refuse every errand on both demo cells forever, which is
    a robot that does nothing wearing a safety feature's clothes. A world
    whose charged pack cannot fund its dearest job plus the return trip is a
    world that never had the margin to spend, and saying so is more honest
    than pretending to enforce one.
    """
    if reserve_wh <= 0.0:
      return 0.0
    return float(reserve_wh) \
        if self.dearest_wh() + reserve_wh <= float(charged_wh) else 0.0

  def afford(self, action: str, *, energy_wh: float, charged_wh: float,
             reserve_wh: float, cost_wh: float | None = None,
             target: str = "") -> Affordability:
    """One of the four states for one errand, right now.

    `cost_wh` overrides the table for an errand that carries its own estimate
    -- a task's, frozen when the offer was made, so that what the gate
    refuses and what the board refused to let anyone claim are the same
    number even if the table is re-measured mid-run.
    """
    cost = float(self.cost(action, target) if cost_wh is None else cost_wh)
    margin = self.margin_wh(charged_wh, reserve_wh)
    need = cost + margin
    if need > float(charged_wh):
      # No charge in this world makes this errand fit. What that MEANS
      # depends on which regime the world is in, and the two are opposite:
      # a world that funds margins and still cannot cover this job is a world
      # where the job is impossible, so it is dropped. A world with no margin
      # to fund is a demo cell, which is smaller than its jobs on purpose --
      # refusing there would delete work the robot demonstrably does.
      state = BEYOND if margin > 0.0 else OVERSPEND
    elif need > float(energy_wh):
      state = CHARGE_FIRST
    else:
      state = OK
    return Affordability(state=state, cost_wh=cost, need_wh=need,
                         have_wh=float(energy_wh),
                         charged_wh=float(charged_wh), action=action)

  # ---- what the overseer is shown --------------------------------------

  def as_context(self, actions=()) -> dict:
    """`{action: Wh}` for the model's prompt, restricted to what it may ask
    for.

    This rides the STABLE prefix (`overseer.system_prompt`), not the volatile
    turn: what an errand costs is a property of the world and does not change
    between calls, so putting it in the user turn would invalidate the cache
    for nothing. What DOES change -- which of them the pack can pay for right
    now -- goes in the volatile half as `affordableActions`.

    ⚠ MEASURED ROWS ONLY. `cost` answers for any name, because the gate has
    to price an unmeasured errand at something -- but printing that fallback
    as though it were a measurement would tell the model that `idle` costs
    0.97 Wh, which is both false and exactly the kind of confident wrong
    number the whole design keeps out of the prompt.
    """
    names = tuple(actions) or tuple(sorted(n for n in self.errand_wh
                                           if ":" not in n))
    return {name: round(self.errand_wh[name], 3) for name in names
            if self.errand_wh.get(name, 0.0) > 0.0}


def load(world: str = "", path: str | os.PathLike | None = None) -> EnergyModel:
  """Module-level convenience, matching `cadence.default_cadence`."""
  return EnergyModel.load(world, path)
