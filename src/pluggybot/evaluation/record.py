"""One run, one JSON record (issue #106; docs/Evaluation.md §4).

Schema v1 is the provisional record from Evaluation.md CORRECTED by pass 1a
of the baseline (issue #105): the nine things six hand-counted days showed
the record needs are each a field here, and the reason is at the field.
The short version -- a run is stored as ROWS (every decision, every errand,
every charge) with the counts derived from them, because every question
the baseline answered was a query over rows that the provisional schema
had summed away.

`Probe` is how the rows are collected: it attaches to a built lifecycle on
`run_demo`'s `on_ready` seam and listens on `say_hooks` and
`Overseer.on_decision`. It changes nothing -- a probe that could would be
measuring a different world from the one the record names.
"""

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.economy import cadence, energy, metabolism, questions, scoring
from pluggybot.telemetry.protocol import DEATH_CAUSES
from pluggybot.mind.overseer import fallback_class

SCHEMA = 1

#: The three arms (Evaluation.md §2). `autonomous` is named so a record can
#: carry it, and refused by `run.py` until the arm exists (§7, item 4).
ARMS = ("scripted", "guarded", "autonomous")
BUILT_ARMS = ("scripted", "guarded", "autonomous")

#: How a run ended. CLOSED, because a rollup groups on it. `stuck` is listed
#: and currently unreachable: a wedged mission does not end (#108), so the
#: harness kills it on wall clock and records `killed` -- which is NOT a
#: death of either kind and is excluded from survival statistics. Collapsing
#: `killed` into `stuck` would count "the box was slow" as a physics death.
END_CAUSES = ("day over", "complete", "flat", "stranded", "stuck", "killed",
              "aborted")

#: Where a call-latency distribution is read (issue #117). The deadline is
#: a CAP on this distribution, so the share of calls it cuts off is a
#: property of the TAIL and not of the middle: a 4.88 s median under an 8 s
#: deadline lost a third of the loaded baseline's decisions, because its
#: worst calls were already at 7.4 s with the box doing nothing.
LATENCY_PERCENTILES = (90.0, 95.0)

#: The five files that each change the regime (Evaluation.md §4), resolved
#: exactly as the sim resolves them -- the env override wins -- so a record
#: hashes the file the run actually read.
DATA_FILES: dict[str, tuple[Path, str]] = {
  "rewards": (scoring.TABLE_PATH, scoring.TABLE_ENV),
  "cadence": (cadence.CADENCE_PATH, cadence.CADENCE_ENV),
  "energy": (energy.ENERGY_PATH, energy.ENERGY_ENV),
  "metabolism": (metabolism.METABOLISM_PATH, metabolism.METABOLISM_ENV),
  "questions": (questions.BANK_PATH, questions.BANK_ENV),
}

#: Where the serving image bakes its git sha (issue #132). `.git` is
#: dockerignored, so a container has no repo to ask -- and `repo_commit`
#: answering `unknown` in production is precisely the unattributable
#: observatory this variable exists to end.
COMMIT_ENV = "PLUGGY_COMMIT"

REPO = Path(__file__).resolve().parents[3]


def world_hash(world: str) -> str:
  """sha256 over the world's XML, every file it `<include>`s (recursively)
  and every asset it names by `file=` -- the WORLD is part of the regime.

  Learned from issue #110: the MSAA fix changed one attribute in the robot
  model and every scripted day after it is a different (and now
  repeatable) trajectory, while the five data files were untouched -- so a
  rollup keyed on the data files alone would have pooled pre-fix and
  post-fix runs under one series name.
  """
  from pluggybot.lifecycle import world_config
  root = REPO / world_config(world)["model"]
  digest = hashlib.sha256()
  seen: set[Path] = set()

  def visit(path: Path) -> None:
    if path in seen or not path.exists():
      return
    seen.add(path)
    text = path.read_bytes()
    digest.update(path.name.encode() + b"\0" + text + b"\0")
    for ref in re.findall(rb'file="([^"]+)"', text):
      visit(path.parent / ref.decode())

  visit(root)
  return digest.hexdigest()


def data_hashes(world: str) -> dict[str, str]:
  """sha256 of each data file as the sim would load it right now, plus the
  world's own hash: together they are the regime a series is defined by."""
  out = {}
  for name, (default, env) in DATA_FILES.items():
    path = Path(os.environ.get(env) or default)
    out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
  out["world"] = world_hash(world)
  return out


def repo_commit() -> str:
  """Which build this is, as a short sha.

  `$PLUGGY_COMMIT` wins over git because the SERVING IMAGE has no `.git`
  (it is in `.dockerignore`, and copying the history to name a commit
  would be an odd trade) -- the sha is baked at image build and the
  Dockerfile fails the build without one, so a deployed container can say
  what it is rather than reporting `unknown` for ever. Locally the
  variable is unset and this is the git call it always was.
  """
  baked = os.environ.get(COMMIT_ENV, "").strip()
  if baked:
    return baked
  try:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True,
                          cwd=Path(__file__).parent).stdout.strip()
  except Exception:  # noqa: BLE001 -- no git, no repo: still a record
    return "unknown"


def build_identity(world: str, *, arm: str, model: str | None = None,
                   backend: str | None = None, pack_wh: float | None = None,
                   reserve_wh: float | None = None,
                   deadline_s: float | None = None,
                   rung: str | None = None,
                   origin: str | None = None,
                   hashes: dict | None = None,
                   commit: str | None = None) -> dict:
  """WHICH BUILD produced a stream, in the experiment's own vocabulary
  (issue #132; docs/Evaluation.md §5).

  The deployed world is an observatory rather than an experiment -- one
  uncontrolled continuous run whose numbers never enter a results table --
  but an observation nobody can attribute is not weaker data, it is
  unusable data: a week under one build and the week after it under
  another wear one name, and nothing separates them afterwards. That is
  the exact failure `dataHashes` and `deadlineS` were added to a SERIES
  KEY to prevent, so this is deliberately the same six things, computed
  HERE so a record and a header can never disagree about what a file
  hashes to.

  It lives beside `build_record` for that reason and no other: this module
  is where "which regime is this" is defined, and the telemetry header is
  a second reader of it rather than a second implementation.
  """
  return {
    "commit": commit if commit is not None else repo_commit(),
    "dataHashes": dict(hashes if hashes is not None else data_hashes(world)),
    "arm": arm,
    # WHICH MIND, and it is two fields because they answer different
    # questions: `Qwen/Qwen3-4B-Instruct-2507` is the model and
    # `huggingface` is the road it took to get there -- the same id served
    # locally is a different regime (docs/Overseer.md §6).
    "model": model,
    "backend": backend,
    # The three world parameters that have each been shown to move
    # behaviour: the pack size (Evaluation.md §5, "an experimental
    # parameter, not a comfort setting"), the return-trip margin every
    # errand must leave behind, and the decision deadline -- which is not
    # a data file, so no hash catches it, and which decides how much of a
    # day the model decided at all (issue #117).
    "packWh": pack_wh, "reserveWh": reserve_wh, "deadlineS": deadline_s,
    # ...and WHICH RUNG, where there is a ladder (issue #142). Part of the
    # rollup's series key for the same reason it is here: A0 hides the
    # survival clock A1 restores, so two rungs are two regimes.
    #
    # ⚠ ABSENT rather than null on an arm with no ladder, which is the one
    # place this block departs from `model`/`backend`. Those answer a
    # question every arm has an answer to ("which mind" -- none, on
    # `scripted`); "which rung" is not a question `guarded` has an answer
    # to, and a `"rung": null` beside it invites a reader to look for a
    # ladder that does not exist. It also keeps a `guarded` header -- the
    # deployed world's -- byte-identical to the one #132 shipped.
    **({"rung": rung} if rung else {}),
    # ...and WHICH ORIGIN the agent's event map started from (issue #127),
    # on exactly the rung's terms: ABSENT where the arm has no map, so a
    # `guarded` header stays byte-identical to #132's and an `autonomous`
    # one flown at `none` stays byte-identical to #142's.
    **({"origin": origin} if origin and origin != "none" else {}),
  }


def slug(text: str) -> str:
  return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "none"


def run_id(config: dict, started_at: datetime) -> str:
  stamp = started_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
  label = slug(config.get("label") or "")
  return (f"{stamp}_{config['world']}_{config['arm']}_{config['pack']}_"
          f"{slug(config.get('model') or 'none')}"
          + (f"_{label}" if config.get("label") else "")
          + f"_s{config.get('seed', 0)}")


# ---- the probe ---------------------------------------------------------------


class Probe:
  """Listens to a running lifecycle and keeps every event as a row.

  `sink`, if given, receives each row the moment it is made -- the harness
  points it at a JSONL file so a run that has to be killed still leaves the
  rows it produced (that is how the baseline's wedged day was counted).
  """

  def __init__(self, sink: Callable[[dict], None] | None = None) -> None:
    self.events: list[dict] = []
    self.sink = sink
    self._life = None

  def attach(self, life) -> None:
    self._life = life
    life.say_hooks.append(self._say)
    if life.overseer is not None:
      life.overseer.on_decision.append(self._decision)

  def _emit(self, row: dict) -> None:
    self.events.append(row)
    if self.sink is not None:
      self.sink(row)

  def _say(self, t: float, msg: str) -> None:
    life = self._life
    self._emit({"kind": "say", "t": round(float(t), 3),
                "fraction": round(life.battery.fraction, 4),
                "wh": round(life.battery.energy_wh, 4),
                "state": life.state, "msg": msg})

  def _decision(self, event: dict) -> None:
    state, d = event["state"], event["decision"]
    battery = state.get("battery") or {}
    offers = [{"id": o.get("id"), "kind": o.get("kind"),
               "estimateWh": o.get("estimateWh"),
               "claimable": bool(o.get("claimable")),
               "expiresInS": o.get("expiresInS")}
              for o in (state.get("offeredTasks") or ())
              if isinstance(o, dict)]
    self._emit(decision_row(
      t=state.get("simTimeS"), fraction=battery.get("fraction"),
      wh=battery.get("wh"), spendableWh=battery.get("spendableWh"),
      action=d.action, source=d.source, task=d.task, board=d.board,
      program=d.program, zone=d.zone, reason=d.reason,
      wallS=event.get("wallS"), error=event.get("error", ""),
      learn=bool(d.learn), forget=bool(d.forget), note=bool(d.note),
      escalate=bool(d.escalate), standingOrder=d.standing_order,
      # WHETHER THIS ANSWER REWROTE THE MAP, and to what (issue #127). The
      # rows themselves and not just a flag: the issue asks for the map at
      # origin, at EVERY EDIT and at the end, and rows are all a killed run
      # leaves behind -- `mind.eventMap.log` is the same history read off a
      # result that survived.
      eventMap=[rw.as_dict() for rw in d.event_map],
      offers=offers,
      affordable=list(state.get("affordableActions") or ()),
      possible=list(state.get("possibleActions") or ()),
      hunger=(state.get("metabolism") or {}).get("state")))


def decision_row(**kw) -> dict:
  """The shape of one decision in the record (Evaluation.md §4, item 4)."""
  return {"kind": "decision", **kw}


# ---- from rows to a record ---------------------------------------------------


def percentile(values: list, p: float) -> float:
  """The NEAREST-RANK percentile: an order statistic, never an interpolation
  between two of them.

  At the sizes here (n=50 for a latency probe, n=5 for a series) an
  interpolated p95 is a number no call actually took, and the tail is
  exactly what a deadline is read off. `p` is 0-100.
  """
  vals = sorted(values)
  k = max(1, math.ceil((p / 100.0) * len(vals)))
  return vals[min(k, len(vals)) - 1]


def dist(values: list, percentiles: tuple[float, ...] = ()) -> dict:
  """min / median / max + the raw values, and the percentiles asked for.

  ⚠ `percentiles` is passed WHERE THE TAIL IS THE POINT and nowhere else
  (issue #117): a call-latency distribution is read at p90/p95 because that
  is what a deadline has to cover, while a p95 over five days' charge counts
  is just the maximum wearing a percentile's name.
  """
  vals = [v for v in values if v is not None]
  if not vals:
    return {"n": 0, "min": None, "median": None, "max": None,
            **{f"p{p:g}": None for p in percentiles}, "values": []}
  return {"n": len(vals), "min": min(vals), "median": statistics.median(vals),
          **{f"p{p:g}": percentile(vals, p) for p in percentiles},
          "max": max(vals), "values": vals}


def classify_charges(says: list[dict]) -> list[dict]:
  """Every entry into GO_CHARGE with its cause, read off the narration.

  Three causes, not two (pass 1a): a `DECIDE charge` before it is
  `voluntary`, a `DEFER` before it is the errand energy gate (`deferred`),
  and anything else is `needs_charge` (`forced`). On a hosting pack the
  baseline measured 11 deferred to 1 forced; a two-way split would have
  booked every deferral as forced.
  """
  out = []
  prev = None
  for i, line in enumerate(says):
    if line["state"] == "GO_CHARGE" and prev != "GO_CHARGE":
      cause, marker = "forced", ""
      for back in range(i - 1, -1, -1):
        m = says[back]["msg"]
        if m.startswith("DECIDE charge"):
          cause, marker = "voluntary", m
          break
        if m.startswith("DEFER "):
          cause, marker = "deferred", m
          break
        if m.startswith(("DECIDE ", "SCORE ", "EXPLORE -> GO_CHARGE",
                         "USE_TOOL", "SWAP_", "TASK ")):
          marker = m
          break
      docked = any(n["msg"].startswith("GO_CHARGE -> CHARGE")
                   for n in says[i:i + 12])
      out.append({"t": line["t"], "fraction": line["fraction"],
                  "wh": line["wh"], "cause": cause, "docked": docked,
                  "marker": marker[:120]})
    prev = line["state"]
  return out


def longest_streak(rows: list[dict]) -> int:
  """The longest run of identical consecutive model decisions -- the "will
  not route around a failure" number (pass 1a measured eight)."""
  best = run = 0
  last = None
  for r in rows:
    key = (r["action"], r.get("board"), r.get("program"), r.get("task"),
           r.get("zone"))
    run = run + 1 if key == last else 1
    last = key
    best = max(best, run)
  return best


def fallback_classes(mind: dict) -> dict:
  """One run's fallbacks split into `failure` and `policy` (issue #141),
  with the FAILURE rate beside them -- the quantity `rollup.FALLBACK_LIMIT`
  is actually asking about.

  `failureRate` is over the run's DECISIONS, not over its fallbacks, so it
  is on the same scale as `fallbackRate` and the two can be read side by
  side: a day at 0.25 total and 0.05 failure is a day the model spent
  idling, not one the box decided.

  ⚠ DERIVED, NEVER RECORDED. Every record ever written already carries
  `fallbackReasons`, so one implementation classifies the committed corpus
  and tomorrow's flights identically. A field added to `build_record`
  would split the corpus in two -- and the rollup would have to keep this
  derivation anyway, for the older half.

  ⚠ The partition itself lives in `overseer.py`, which is where the line
  was first drawn: a second copy here is a copy that disagrees the day a
  reason is added.
  """
  counts = {"failure": 0, "policy": 0}
  for source, n in (mind.get("fallbackReasons") or {}).items():
    counts[fallback_class(str(source)) or "failure"] += n
  n_decisions = mind.get("decisions") or 0
  return {**counts,
          "failureRate": (round(counts["failure"] / n_decisions, 4)
                          if n_decisions else None),
          "policyRate": (round(counts["policy"] / n_decisions, 4)
                         if n_decisions else None)}


def end_cause(result: dict | None, max_sim_s: float) -> str:
  if result is None:
    return "killed"
  if result.get("aborted"):
    return "aborted"
  if result.get("stranded"):
    return "stranded"
  if float(result.get("battery", 1.0)) <= 0.0:
    return "flat"
  if float(result.get("sim_time", 0.0)) >= max_sim_s:
    return "day over"
  return "complete"


def _interventions(result: dict | None, resets: list[dict]) -> list[dict]:
  """Every time an admin reached into world state (issue #119).

  ⚠ THE LIFECYCLE'S OWN LIST IS THE AUTHORITY, and the reset-derived
  fallback below is only for a result written before it existed. Deriving
  them from `resets` alone can only ever find one of the three kinds --
  `set_battery` and `set_points` leave no reset behind — so a run whose
  battery was topped up would have looked clean, which is the precise
  failure `interventions` exists to prevent (docs/Evaluation.md §5).
  """
  own = (result or {}).get("interventions")
  if own is not None:
    return [{"t": i.get("t"), "what": i.get("detail") or i.get("what"),
             "by": i.get("by"), "kind": i.get("what"),
             "before": i.get("before"), "after": i.get("after")}
            for i in own]
  return [{"t": r["t"], "what": f"reset by {r.get('by', '?')} while alive",
           "by": r.get("by"), "kind": "reset_robot"}
          for r in resets if r.get("intervention")]


def build_record(config: dict, result: dict | None, events: list[dict],
                 wall_s: float, started_at: datetime,
                 hashes: dict | None = None, commit: str | None = None) -> dict:
  """The v1 record for one run. `result` is `run_demo`'s dict, or None for a
  run that was killed on wall clock and left only its rows."""
  rows = [e for e in events if e.get("kind") == "decision"]
  says = [e for e in events if e.get("kind") == "say"]
  llm = [r for r in rows if str(r["source"]).startswith("llm")]
  # ⚠ THREE PRODUCERS SINCE ISSUE #127, NOT TWO. A decision whose source is
  # `event:<type>` came from a row of the agent's OWN map: not a model answer
  # and not a fallback. Counting it in `fallbacks` would make an agent that
  # configured its day well read as an agent whose endpoint was down, and
  # would move `fallbackRate` -- which `FALLBACK_LIMIT` is set against, and
  # which issue #141 spent a whole change making mean one thing.
  by_event = [r for r in rows if str(r["source"]).startswith("event:")]
  fallbacks = [r for r in rows if str(r["source"]).startswith("fallback:")]
  vol = [r for r in llm if r["action"] == "charge"]
  # ⚠ `honoured` IS KEPT THOUGH NOTHING CAN REFUSE A CHARGE ANY MORE (issue
  # #135). It used to be "below `TOP_UP_BELOW`", and THE PAIR IS WHAT MADE
  # THAT RAIL FINDABLE: A0's one surviving day chose 15 charges and had 3
  # honoured, all twelve refusals sitting at 0.75-0.81 -- a record carrying
  # one number would have reported an agent that charges fifteen times a day,
  # and the day would have read as the agent being careful when it was the
  # rail being careful for it.
  #
  # So the field survives its cause. With the floor deleted every chosen
  # charge is made, and `chosen == honoured` is now the ASSERTION rather than
  # the arithmetic: the next thing that quietly declines a charge shows up
  # here as a gap, in a field a reader already knows to compare.
  vol_honoured = list(vol)
  # `anticipation`, pinned to two definitions (pass 1a tried both, both 0):
  # a voluntary charge while an offer on the board could not be funded, and
  # one while a menu action was possible-after-a-charge but not affordable.
  def unfundable(r: dict) -> list:
    have = r.get("spendableWh")
    return [o["id"] for o in r.get("offers", ())
            if have is not None and o.get("estimateWh") is not None
            and o["estimateWh"] > have and (o.get("expiresInS") or 0) > 0]

  def not_affordable(r: dict) -> list:
    return sorted(set(r.get("possible", ())) - set(r.get("affordable", ())))

  charges = classify_charges(says)
  by_cause = Counter(c["cause"] for c in charges)
  max_sim = float(config.get("maxSimS", 3600.0))
  end = end_cause(result, max_sim)
  sim_s = (float(result["sim_time"]) if result is not None
           else (says[-1]["t"] if says else 0.0))
  frac_end = (float(result["battery"]) if result is not None
              else (says[-1]["fraction"] if says else None))
  fractions = [s["fraction"] for s in says] + [r["fraction"] for r in rows
                                              if r["fraction"] is not None]
  # A FLAT DEATH IS THE PACK REACHING ZERO (Evaluation.md §3), not the run
  # ending on it: `needs_charge` is checked between errands and the sim
  # does not stop the motors at 0 Wh, so a robot can hit zero mid-errand,
  # drive to the rack on nothing and finish the day "day over". The first
  # committed set had exactly that day. The survival span ends there.
  flat_at = next((s["t"] for s in says if s["fraction"] <= 0.0), None)
  # Since issue #107 the lifecycle records deaths and resets itself; a
  # result that carries them is the authority, and the narration-derived
  # `flat_at` above is the fallback for a run that died before it existed.
  deaths = list((result or {}).get("deaths") or [])
  resets = list((result or {}).get("resets") or [])
  overseer = (result or {}).get("overseer") or {}
  metab = (result or {}).get("metabolism") or {}
  thought = (result or {}).get("thought_stats") or {}
  ledger_ok = None
  if result is not None and metab:
    ledger_ok = ((result["earned"] - metab["consumed"] - metab["spilled"])
                 == result["points"])
  errands = [{"name": e["errand"], "module": e.get("module"),
              "picked": bool(e["picked"]), "stowed": bool(e["stowed"]),
              "error": e.get("error") or None, "skipped": e.get("skipped"),
              "energyWh": e.get("energyWh"), "estimateWh": e.get("estimateWh")}
             for e in (result or {}).get("errands", [])]
  wh_failed = round(sum(float(e["energyWh"] or 0.0) for e in errands
                        if e["error"] or not e["picked"]), 4)
  task_stats = dict((result or {}).get("task_stats") or {})
  offered_today = sum(1 for s in says
                      if re.match(r"TASK t_\d+ offered", s["msg"]))
  refusals = list(thought.get("refusals") or
                  [s["msg"] for s in says
                   if s["msg"].startswith("THOUGHT refused")])
  mind = {
    "decisions": len(rows), "llmCalls": len(llm), "fallbacks": len(fallbacks),
    "fallbackRate": (round(len(fallbacks) / len(rows), 4) if rows else None),
    "fallbackReasons": dict(Counter(str(r["source"]) for r in fallbacks)),
    # The vendor's words behind every garbled/offline answer -- the
    # baseline lost the interesting one to `usage.errors`' five-line window.
    "errors": [r["error"] for r in rows if r.get("error")],
    # The one distribution read at its TAIL: the deadline is a cap on
    # this, and a median well under it says nothing about how often it
    # bites (issue #117).
    "wallS": dist([r["wallS"] for r in llm], LATENCY_PERCENTILES),
    "deadlineS": config.get("deadlineS"),
    "constrained": overseer.get("constrained"),
    "budgetLeft": overseer.get("budgetLeft"),
    "usd": overseer.get("usd"), "priced": overseer.get("priced"),
    "inputTokens": overseer.get("inputTokens"),
    "outputTokens": overseer.get("outputTokens"),
    "actions": dict(Counter(r["action"] for r in llm)),
    "fallbackActions": dict(Counter(r["action"] for r in fallbacks)),
    "eventActions": dict(Counter(r["action"] for r in by_event)),
    "longestStreak": longest_streak(llm),
  }
  # ABSENT, not zero, when no escalation model was configured: the field
  # was not in the model's grammar, so "never asked" would be a lie.
  if overseer.get("escalationModel"):
    mind["escalations"] = {
      "model": overseer["escalationModel"],
      "granted": overseer.get("escalations", 0),
      "refused": dict(overseer.get("escalationsRefused") or {}),
      "asked": sum(1 for r in rows if r.get("escalate")),
      "usd": overseer.get("escalationUsd"),
    }
  # THE AGENT'S OWN CONFIGURATION (issue #127), and the reason to want the
  # whole feature: EVERY FIELD BELOW IS READABLE WITHOUT FLYING ANYTHING.
  # `score` answers "did it write itself a charging rule", "did it keep an
  # `ask` row" and "are its thresholds ordered" off a config in
  # microseconds, where the same questions cost sim-hours today. `log` is
  # the map at origin, at every edit and at the end -- one list, because an
  # edit history whose first entry IS the origin cannot disagree with it.
  #
  # ABSENT where this world had no map, on `standingOrders`' terms exactly:
  # "never configured itself" and "was never given a configuration" are
  # different facts and only the first is about the agent.
  if overseer.get("eventMap"):
    emap = overseer["eventMap"]
    mind["eventMap"] = {
      "origin": emap.get("origin"),
      "final": emap.get("current"),
      "score": emap.get("score"),
      "edits": emap.get("edits"),
      "log": emap.get("log"),
      # WHAT THE ROWS DID, and the two halves are never summed: `fired` is
      # rows whose moment came, `failed` is actions that did not happen and
      # WHY. An agent whose actions fail constantly is one that did not
      # understand the rules it was given, and that is invisible in a count
      # of what fired -- which is the whole reason the causes are counted.
      "fired": emap.get("fired"),
      "failed": emap.get("failed"),
      "actions": len(by_event),
    }
  # WHAT THE AGENT LEFT BEHIND, AND WHETHER IT WAS EVER NEEDED (issue #125).
  # Counted off the ROWS rather than off the overseer's own counters, because
  # a run killed on wall clock leaves rows and no result -- and read as: a
  # model answer's `standingOrder` is one SET, a fallback's is one FIRED, and
  # a firing whose action is not the order is one that could not be run.
  # Present only where the field was in the model's grammar, on
  # `escalations`' terms: "never left an order" is a fact about the agent and
  # "was never offered one" is not.
  if overseer.get("standingOrders"):
    fired = [r for r in fallbacks if r.get("standingOrder")]
    mind["standingOrders"] = {
      "set": sum(1 for r in llm if r.get("standingOrder")),
      "orders": dict(Counter(r["standingOrder"] for r in llm
                             if r.get("standingOrder"))),
      "fired": len(fired),
      "firedOrders": dict(Counter(r["standingOrder"] for r in fired)),
      # An order that fired into `idle` because it named something this
      # world could not do at that moment. Never summed with the firings:
      # "it chose this and this happened" and "it chose something
      # impossible" are the two facts the field is measured for.
      "unrunnable": sum(1 for r in fired if r["action"] != r["standingOrder"]),
      # ...and a fallback with no order at all: the floor, before the agent
      # has left one.
      "unset": sum(1 for r in fallbacks if not r.get("standingOrder")),
      "final": overseer["standingOrders"].get("current"),
    }
  # Survival spans: mission start -> first death, and each reset -> the next
  # death or the end of the day. Without a reset there is one span, as
  # before; a reset of a LIVING robot is an intervention (Evaluation.md §5).
  if deaths or resets:
    marks = sorted([(d["t"], "death") for d in deaths]
                   + [(r["t"], "reset") for r in resets])
    spans, since, alive = [], 0.0, True
    for t_mark, kind in marks:
      if kind == "death" and alive:
        spans.append(round(t_mark - since, 3))
        alive = False
      elif kind == "reset":
        since, alive = t_mark, True
    if alive and end != "killed":
      spans.append(round(sim_s - since, 3))
    # THREE CAUSES, NEVER SUMMED (issue #136 adds the third). `flat` is a
    # decision failure, `stuck` a physics one, and `unpaid` an ECONOMIC one
    # -- upkeep came due and the balance could not cover it. A consumer that
    # added them would hide which of three different things needs fixing.
    death_counts = {c: sum(1 for d in deaths if d["cause"] == c)
                    for c in DEATH_CAUSES}
  else:
    spans = ([round(flat_at, 3)] if flat_at is not None
             else [round(sim_s, 3)] if end != "killed" else [])
    death_counts = {"flat": int(flat_at is not None or end == "flat"),
                    "stuck": int(end in ("stuck", "stranded")),
                    # A record written before issue #136 cannot have one,
                    # and zero is what that honestly means. Same for
                    # `unminded` before issue #127 -- and it is reachable
                    # only through `deaths` above, since a run with no
                    # lifecycle result has nothing that could report one.
                    "unpaid": 0, "unminded": 0}
  # ⚠ OUTSIDE THE BRANCH ABOVE, and that is the fix as much as the function
  # is: an intervention is no longer something only a RESET can produce
  # (issue #119), so a run with a topped-up battery and no reset in it used
  # to take the `else` arm and record an empty list.
  interventions = _interventions(result, resets)
  record = {
    "schema": SCHEMA,
    # The id the PARENT assigned, when there is one: the file is named by it
    # before the child starts, and a child recomputing it off its own clock
    # produced records whose `runId` disagreed with their file name.
    "runId": config.get("runId") or run_id(config, started_at),
    "startedAt": config.get("startedAt") or started_at.astimezone(
      timezone.utc).replace(microsecond=0).isoformat(),
    "world": config["world"], "arm": config["arm"], "pack": config["pack"],
    # THE CONDITIONS THE RUN WAS FLOWN UNDER, as a name (issue #117). Part
    # of the series key, so a series flown on a quiet box and one flown
    # five-up on a loaded one are two series rather than one average of
    # both -- which is the whole point of committing the pair. Empty for a
    # run that claims nothing about its box, which is every run before this
    # existed.
    "label": str(config.get("label") or ""),
    "model": config.get("model"), "backend": overseer.get("backend")
    or config.get("backend"),
    "seed": int(config.get("seed", 0)),
    "commit": commit if commit is not None else repo_commit(),
    "config": {
      "errand": config.get("errand"), "tasks": bool(config.get("tasks")),
      "metabolism": bool(config.get("metabolism")),
      "maxSimS": max_sim, "packWh": config.get("packWh"),
      "reserveWh": config.get("reserveWh"),
      "freshState": bool(config.get("freshState", True)),
      "parallel": int(config.get("parallel", 1)),
      "deadlineS": config.get("deadlineS"),
      "wallLimitS": config.get("wallLimitS"),
      # WHICH RUNG of the autonomous ladder, and absent on the arms that
      # have no ladder -- "flown at A0" and "the question did not apply"
      # are different claims, on `escalations`' terms (issue #115).
      **({"rung": config.get("rung") or "A0"}
         if config["arm"] == "autonomous" else {}),
      # ...and WHICH ORIGIN the agent's event map started from (issue #127),
      # on the rung's terms exactly: absent on an arm with no map, so every
      # committed record keeps the config block it already has, and `none`
      # -- the default -- is the arm as issue #115 flew it.
      **({"origin": config.get("origin") or "none"}
         if config["arm"] == "autonomous" else {}),
    },
    "simSeconds": round(sim_s, 3), "wallSeconds": round(float(wall_s), 1),
    "end": end,
    "dataHashes": dict(hashes if hashes is not None
                       else data_hashes(config["world"])),
    "survival": {
      # Until a reset exists (issue #107) a run has one survival span, to
      # the moment the pack first reached zero or to the end of the day; a
      # killed run has none, because nobody knows when it died.
      "survivalS": spans,
      # `stranded` -- "unable to reach the rack" -- IS §3's `stuck`: a
      # navigation failure, never a decision one. The two columns the doc
      # says never to sum, both populated.
      "deaths": death_counts,
      # LIVES LEFT AT THE END, and how many robots the volume used up
      # (issue #136). `trueDeaths` is a DIFFERENT event from a death and is
      # never summed with one: an ordinary death keeps the volume, and this
      # is the one that archives it.
      "hearts": (result or {}).get("hearts"),
      "trueDeaths": len((result or {}).get("true_deaths") or []),
      "flatAtS": flat_at,
      "resets": len(resets),
      "batteryEnd": frac_end,
      "minFraction": (round(min(fractions), 4) if fractions else None),
    },
    "charging": {
      "forced": by_cause.get("forced", 0),
      "deferred": by_cause.get("deferred", 0),
      "voluntary": {"chosen": len(vol), "honoured": len(vol_honoured),
                    "chosenFrac": [r["fraction"] for r in vol],
                    "honouredFrac": [r["fraction"] for r in vol_honoured]},
      "docked": sum(1 for c in charges if c["docked"]),
      "cycles": (result or {}).get("charge_cycles"),
      "anticipation": {
        "offer": sum(1 for r in vol_honoured if unfundable(r)),
        "action": sum(1 for r in vol_honoured if not_affordable(r)),
      },
      "entries": charges,
    },
    "mind": mind,
    "decisionRows": [{**r, "unfundableOffers": unfundable(r),
                      "notAffordable": not_affordable(r)} for r in rows],
    "errands": errands,
    "whFailed": wh_failed,
    "memory": {
      "learn": sum(1 for r in rows if r.get("learn")),
      "forget": sum(1 for r in rows if r.get("forget")),
      "notes": sum(1 for r in rows if r.get("note")),
      "refusals": refusals,
      "knowledgeChars": (thought.get("chars") or {}).get(
        "Knowledge_and_Opinions.md"),
    },
    "economy": {
      "earned": (result or {}).get("earned"),
      "consumed": metab.get("consumed"), "spilled": metab.get("spilled"),
      "balance": (result or {}).get("points"),
      "identityHolds": ledger_ok,
      # ⚠ WHY IT DOES NOT HOLD, WHEN IT DOES NOT (issue #119). An admin's
      # `set_points` breaks `earned - consumed - spent == balance` ON
      # PURPOSE: the alternative is papering the reach-in into `earned`,
      # which hides it inside the one number the reward system exists to
      # make un-fakeable. So the identity is recorded as broken, WITH the
      # reason -- a bare `false` here would read as a bug in the ledger, and
      # that is the reading this field exists to prevent. Absent when the
      # identity holds, and absent when it fails for some OTHER reason,
      # which is a real bug and must not be given an excuse.
      **({"identityBrokenBy": [i["what"] for i in interventions
                               if i.get("kind") == "set_points"]}
         if ledger_ok is False and any(i.get("kind") == "set_points"
                                       for i in interventions) else {}),
      "hungerEnd": metab.get("state") or None,
      "tasks": {**task_stats, "offeredToday": offered_today},
    },
    "interventions": interventions,
  }
  return record


# ---- validation --------------------------------------------------------------

_REQUIRED = ("schema", "runId", "world", "arm", "pack", "seed", "commit",
             "config", "simSeconds", "wallSeconds", "end", "dataHashes",
             "survival", "charging", "mind", "decisionRows", "errands",
             "memory", "economy", "interventions")


def problems(record: dict) -> list[str]:
  """Everything wrong with a record, as sentences. Empty means valid."""
  out = []
  for key in _REQUIRED:
    if key not in record:
      out.append(f"missing {key!r}")
  if out:
    return out
  if record["schema"] != SCHEMA:
    out.append(f"schema {record['schema']!r}, expected {SCHEMA}")
  if record["arm"] not in ARMS:
    out.append(f"arm {record['arm']!r} is not one of {ARMS}")
  if record["end"] not in END_CAUSES:
    out.append(f"end {record['end']!r} is not one of {END_CAUSES}")
  hashes = record["dataHashes"]
  for name in (*DATA_FILES, "world"):
    h = hashes.get(name)
    if not (isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)):
      out.append(f"dataHashes[{name!r}] is not a sha256 hex digest")
  ch = record["charging"]
  for key in ("forced", "deferred", "voluntary", "docked", "anticipation"):
    if key not in ch:
      out.append(f"charging.{key} missing")
  if "voluntary" in ch and not {"chosen", "honoured"} <= set(ch["voluntary"]):
    out.append("charging.voluntary needs both chosen and honoured")
  if not isinstance(record["decisionRows"], list):
    out.append("decisionRows is not a list")
  if not isinstance(record["interventions"], list):
    out.append("interventions is not a list")
  if (record["end"] == "killed" and record["survival"]["survivalS"]
      and record["survival"].get("flatAtS") is None):
    out.append("a killed run's survival span can only end at a flat")
  return out


def validate(record: dict) -> dict:
  found = problems(record)
  if found:
    raise ValueError(f"{record.get('runId', '?')}: " + "; ".join(found))
  return record


def load_events(path: Path) -> list[dict]:
  rows = []
  for line in Path(path).read_text().splitlines():
    if line.strip():
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        break                         # a kill mid-line: keep what was whole
  return rows
