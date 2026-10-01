"""What an ARM means, in ONE place (Evaluation.md §2; issue #142).

An arm is a claim about who is deciding, and it has to be true wherever it
runs: `scripted` is the loop with no mind, its rails its own, and
`autonomous` is the one mind, with the rails off. The deployed world serves
one (`scripts/serve.py`) and says which in its header, off this module: one
definition, so a stream cannot claim an arm that was not flown.
"""

from pluggybot.mind.events import DEFAULT_ORIGIN, ORIGINS

#: The arms that are built (Evaluation.md §2). `guarded`, the control, was
#: retired in #427: its results were the rover's, and nothing flies it.
BUILT_ARMS = ("scripted", "autonomous")

#: `$PLUGGY_ARM` / `$PLUGGY_RUNG` -- how the served image is told, since the
#: deployment configures the sim with `environment:` alone. Named here rather
#: than in `serve.py` so the module that knows what an arm IS also owns how
#: one is asked for.
ARM_ENV = "PLUGGY_ARM"
RUNG_ENV = "PLUGGY_RUNG"
#: ...and which map the agent starts with (issue #127).
ORIGIN_ENV = "PLUGGY_ORIGIN"

#: The `autonomous` ladder (Evaluation.md §2). One arm, one change per rung,
#: held fixed within a run -- WHICH RUNG FIRST PRODUCES A VOLUNTARY CHARGE is
#: the finding, so the rung is recorded and never inferred.
#:
#: ⚠ A0 HAS TO HIDE THE SURVIVAL CLOCK TO BE THE NULL IT IS DESCRIBED AS.
#: `survival.aliveS` and `survival.deaths` have been in every world's context
#: since issue #107, so an A0 that simply left them there would already BE
#: A1, and "does seeing the stake change anything" could never be asked --
#: no rung would ever have been flown without it.
RUNGS = {
  "A0": {"show_survival": False},
  "A1": {"show_survival": True},
}

#: The rung an `autonomous` run takes when nobody named one. A0 is the null,
#: so it is what "autonomous" means unqualified.
DEFAULT_RUNG = "A0"

#: Which arms have a ladder at all. `--rung` on any other arm is refused
#: rather than ignored: a flag that silently does nothing is how somebody
#: comes to believe they flew A1.
LADDER_ARMS = ("autonomous",)

#: ...and which arms have an ORIGIN (issue #127): a starting event map, and
#: whether it was `seeded` with today's loop re-expressed as rows or left
#: `unseeded` for the agent to write from nothing.
#:
#: ⚠ AN ABLATION, NOT A RUNG, and `none` is the DEFAULT. A rung is one
#: change to what the model is shown; an origin changes what the model is
#: shown AND what it starts with AND -- for `unseeded` -- the prompt, so it
#: carries an ablation's asymmetry (Evaluation.md section 3): a null is
#: strong evidence and a difference is weak. Defaulting to `none` keeps A0
#: and A1 the ladder they were defined as.
ORIGIN_ARMS = ("autonomous",)


def arm_flags(arm: str, rung: str = DEFAULT_RUNG,
              origin: str = DEFAULT_ORIGIN) -> dict:
  """What an arm means to `serve.py` (issue #142): whether there is a mind,
  and on the one mind which map it starts with and which rung it is on. An
  arm that is not built is refused rather than silently run as another: a
  record claiming an arm that was not flown is the worst kind of result.

  ⚠ THERE IS NOTHING ELSE TO SET (issue #427). The rails come off wherever
  there is a mind (`HubLifecycle.autonomous` reads the overseer's presence),
  the prompt is the one that says so, and the fallback is the agent's own
  standing order or event map -- every one of them a property of a mind, so
  none of them is a flag an arm could get wrong.
  """
  if arm == "scripted":
    return {"overseer": False}
  if arm == "autonomous":
    if rung not in RUNGS:
      raise ValueError(f"unknown rung {rung!r}; the ladder is "
                       f"{', '.join(sorted(RUNGS))} (Evaluation.md §2)")
    if origin not in ORIGINS:
      raise ValueError(f"unknown origin {origin!r}; the origins are "
                       f"{', '.join(ORIGINS)} (Evaluation.md §2)")
    return {"overseer": True, "origin": origin, **RUNGS[rung]}
  raise NotImplementedError(
    f"arm {arm!r} is not built; the built arms are {BUILT_ARMS} "
    "(docs/Evaluation.md §2)")


def rung_for(arm: str, rung: str | None) -> str | None:
  """The rung this run is actually on, or None where there is no ladder.

  ⚠ NAMING A RUNG FOR AN ARM THAT HAS NONE IS AN ERROR, not a no-op. A0 and
  A1 differ by whether the robot can see its own survival clock, and somebody
  who typed `--arm scripted --rung A1` believes they changed something.
  """
  if arm in LADDER_ARMS:
    return rung or DEFAULT_RUNG
  if rung:
    raise ValueError(
      f"arm {arm!r} has no ladder, so --rung {rung!r} would do nothing; "
      f"rungs belong to {', '.join(LADDER_ARMS)} (Evaluation.md §2)")
  return None


def origin_for(arm: str, origin: str | None) -> str | None:
  """The origin this run is actually on, or None where the arm has none.

  ⚠ NAMING ONE FOR AN ARM THAT HAS NONE IS AN ERROR, `rung_for`'s rule
  exactly: `--arm scripted --origin unseeded` is somebody who believes they
  changed something, and a flag that silently does nothing is how a series
  comes to be described as an ablation nobody ran.
  """
  if arm in ORIGIN_ARMS:
    return origin or DEFAULT_ORIGIN
  if origin and origin != DEFAULT_ORIGIN:
    raise ValueError(
      f"arm {arm!r} has no event map, so --origin {origin!r} would do "
      f"nothing; origins belong to {', '.join(ORIGIN_ARMS)} "
      "(Evaluation.md §2)")
  return None
