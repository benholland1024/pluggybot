#!/usr/bin/env python
"""Ladder B of issue #264, on legs (#407): does a model-equipped robot find a
feature? (Evaluation.md §7.)

A LOCAL flight of the served pair on the `autonomous` arm -- the deployed
prompt and model (`$PLUGGY_MODEL`, else `DEPLOYED_MODEL`, through the
router with `$HF_TOKEN`; `--model` names another), origin `unseeded`,
hosting packs -- with the feature's job
put up as the cadence puts it up, standing the whole flight, and the
feature's phrase in the first robot's inbox at the start as a visitor's
message (the rover's phrases, `rover-final`'s `evaluation/record.py`). The
run is read into one word, the first that applies:

  used      the feature's own grader says it worked (the job done and paid)
  errored   an attempt failed in the world (a procedure cut short, the job
            failed, a wrong finding)
  refused   the robot's own writing was refused (a `define`)
  garbled   the answers were garbled
  declined  the robot declined out loud (its reply is printed)
  silence   none of these

Never a test and never a result -- what a model feels like writing that
day cannot fail for a regression reason -- so the reading is a comment on
the issue it informs. Ladder A (`scripts/solve.py`) is the other half: a
solution exists.

  set -a; . ~/pluggybot/.env; set +a
  MUJOCO_GL=egl uv run python scripts/ladder_b.py --feature tower
  MUJOCO_GL=egl uv run python scripts/ladder_b.py --feature bench --sim-s 3600 --out bench.json
"""

import argparse
import json
import os
import tempfile
import time

from pluggybot.lifecycle import QUAD_HOME, task_producer
from pluggybot.mind.inbox import Inbox
from pluggybot.pair import build_pair, run_pair

#: The phrase each feature is asked with, and the job it is about.
PHRASES = {
  "tower": "Please consider trying the tower with a procedure you write.",
  "bench": "Please consider recording a finding at the bench.",
  "census": "Please consider counting the plants in the garden.",
}
JOBS = {"tower": ("stack_tower", "workshop"), "bench": ("find_mass", "lab_bench"),
        "census": ("count_plants", "garden")}
SENDER = "Ben"
#: The deployed pick (#225; the website repo's compose sets `$PLUGGY_MODEL`
#: to it), flown when this environment names none.
DEPLOYED_MODEL = "zai-org/GLM-5.3-Flash:cheapest"


def read(feature: str, job: str, rows: list, decisions: list, board) -> dict:
  """The run as one word, and every count it was read from."""
  of = lambda kind, **m: [r for r in rows if r.get("type") == kind  # noqa: E731
                          and all(r.get(k) == v for k, v in m.items())]
  task = board.get(job)
  reply = next((r for r in of("visitor_reply") if r.get("id") == "probe-1"), None)
  signals = {
    "jobState": task.state if task is not None else None,
    "jobBy": task.claimed_by if task is not None else None,
    "jobReason": (task.verdict or {}).get("reason") if task is not None else None,
    "defined": len(of("procedure", outcome="defined")),
    "defineRefused": len(of("procedure", outcome="refused")),
    "ran": len(of("procedure", outcome="ran")),
    "aborted": len(of("procedure", outcome="aborted")),
    "findings": len(of("finding")),
    "findingsCorrect": len(of("finding", correct=True)),
    "decisions": len(decisions),
    "garbled": sum(1 for d in decisions if d.source == "fallback:garbled"),
    "fallbacks": sum(1 for d in decisions if d.source.startswith("fallback:")),
  }
  done = signals["jobState"] == "done"
  if feature == "bench":
    used = signals["findingsCorrect"] > 0
    errored = signals["findings"] > signals["findingsCorrect"] or signals["aborted"] > 0
  else:
    used = done
    errored = signals["jobState"] == "failed" or signals["aborted"] > 0
  outcome = ("used" if used else "errored" if errored
             else "refused" if signals["defineRefused"] else "garbled" if signals["garbled"]
             else "declined" if reply is not None and reply.get("outcome") == "declined"
             else "silence")
  return {"feature": feature, "outcome": outcome, "signals": signals,
          "reply": ({"outcome": reply.get("outcome"), "text": reply.get("reply", "")}
                    if reply is not None else None)}


def fly(feature: str, sim_s: float, model: str | None) -> dict:
  kind, target = JOBS[feature]
  with tempfile.TemporaryDirectory() as root:
    inboxes = (Inbox(), Inbox())
    lives = build_pair(QUAD_HOME, pack="hosting", overseer=True, origin="unseeded",
                       thoughts_root=f"{root}/thoughts", ledger_state=f"{root}/ledger.json",
                       tasks=True, task_state=f"{root}/tasks.json", mortal=True,
                       inboxes=inboxes, near_field=True, start_points=200,
                       overseer_kw={"model": model} if model else None)
    first = lives[0]
    rows: list = []
    decisions: list = []
    for life in lives:
      life.on_event.append(lambda e, who=life.root: rows.append({"robot": who, **e}))
      life.visitor_hooks.append(lambda e, who=life.root: rows.append({"robot": who, **e}))
      life.overseer.on_decision.append(lambda e: decisions.append(e["decision"]))
    # ...the job up as the cadence puts it up, standing the whole flight
    maker = task_producer(first.tasks, QUAD_HOME, procedures=True,
                          robots=[life.robot_name for life in lives])
    params, secret = maker._build(kind, target)
    job = first.tasks.offer(kind, target, params=params, secret=secret, ttl=sim_s, t=0.0)
    landed = inboxes[0].offer({"type": "message", "id": "probe-1", "from": SENDER,
                               "text": PHRASES[feature]}, t=0.0, sender=SENDER)
    print(f"== ladder B: {feature} ({kind} on {target}, {job.id}); phrase "
          f"{'landed' if landed else 'DROPPED'} in {first.robot_name}'s inbox; "
          f"model {first.overseer.model}; {sim_s:.0f} sim s", flush=True)
    t0 = time.time()

    def settled(lives) -> bool:
      t = first.tasks.get(job.id)
      return t is not None and t.state in ("done", "failed", "expired")
    try:
      run_pair(lives, max_sim_time=sim_s, stop_when=settled)
    finally:
      for life in lives:
        life.body.close()
    out = read(feature, job.id, rows, decisions, first.tasks)
    out.update(simS=round(float(first.data.time), 1), wallS=round(time.time() - t0, 1),
               model=first.overseer.model, phrase=PHRASES[feature])
    return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--feature", required=True, choices=sorted(PHRASES))
  ap.add_argument("--sim-s", type=float, default=2400.0,
                  help="the flight's sim seconds, at most (it ends once the job settles)")
  ap.add_argument("--model", default=os.environ.get("PLUGGY_MODEL") or DEPLOYED_MODEL,
                  help="the mind's model; default $PLUGGY_MODEL, else the deployed pick")
  ap.add_argument("--out", default=None, help="write the reading here as JSON")
  args = ap.parse_args()
  out = fly(args.feature, args.sim_s, args.model)
  print(f"\nLADDER B {out['feature']} -> {out['outcome'].upper()} "
        f"({out['simS']:.0f} sim s, {out['wallS']:.0f} s wall)")
  print(json.dumps(out["signals"], indent=1))
  if out["reply"]:
    print(f"reply ({out['reply']['outcome']}): {out['reply']['text']}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(out, f, indent=2)


if __name__ == "__main__":
  main()
