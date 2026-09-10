#!/usr/bin/env python
"""Run one configuration N times and roll the records up (issue #106).

    MUJOCO_GL=egl uv run python scripts/experiment.py \\
        --arm guarded --world home --pack hosting -n 5
    MUJOCO_GL=egl uv run python scripts/experiment.py --arm scripted -n 5
    uv run python scripts/experiment.py --rollup      # re-aggregate results/

Each run is a child process (`python -m pluggybot.evaluation.run`) with a
fresh state directory, its own record `results/<runId>.json`, and a WALL
CLOCK LIMIT: a mission that wedges never ends on its own (#108), and the
baseline's first attempt at this sat at 0 % for 700 sim-seconds past the
day's budget before a human noticed. A run past the limit is killed and
recorded from the rows it had flushed, with `end: "killed"` -- which the
rollup keeps OUT of the survival statistics, because it measured the box.

`--parallel` runs several at once. Say so in the record (it does): the
baseline's fallback rate was a measurement of five sims sharing six cores.
`--label quiet` says the rest of it -- the box the runs had to themselves --
and puts the two series side by side in the rollup rather than averaging a
loaded day with a quiet one (issue #117). The rollup also DISQUALIFIES a run
whose FAILURE-class fallback rate says the box decided too much of its day
(issue #141 -- the policy class never disqualifies), exactly as it
already drops a killed run and one an admin touched; the run stays committed
and the summary prints why it was dropped.

The rollup refuses, out loud, to aggregate two runs whose data-file hashes
differ; `--rollup` alone re-aggregates whatever is in the results directory
without flying anything, which is what to do after editing a data file (the
committed rollup's `current` flags then flip and `tests/test_experiment.py`
asks for exactly this command).

The `guarded` arm needs the overseer's credentials in the environment
(`$HF_TOKEN` for an `org/name` model, `$ANTHROPIC_API_KEY` for a bare id):
without them every decision is a `fallback:no-client` and the run measures
nothing about a model. The script checks before flying.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from pluggybot.evaluation.record import (
  ARMS, build_record, data_hashes, load_events, run_id, validate,
)
from pluggybot.evaluation.arms import origin_for
from pluggybot.evaluation.rollup import write_rollup
from pluggybot.mind.events import DEFAULT_ORIGIN as ORIGIN_DEFAULT
from pluggybot.mind.events import ORIGINS
from pluggybot.mind import llm

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
#: Wall seconds allowed per sim second before a run is killed. The
#: baseline measured 0.6-1.2x real time on the dev box under load; 3x is a
#: run that has stopped, not a slow one.
WALL_PER_SIM_S = 3.0
#: ...and a FLOOR, because a run has a fixed start-up cost (model load,
#: the detector, the first offscreen render) that a short day is mostly
#: made of: the 30 s test day was killed at 3 x 30 = 90 s under the full
#: suite's load, and reported as `killed` a run that was merely starting.
WALL_FLOOR_S = 600.0


def wall_limit_for(max_sim_s: float, override: float | None = None) -> float:
  if override is not None:
    return float(override)
  return max(WALL_FLOOR_S, WALL_PER_SIM_S * float(max_sim_s))


def credentials_missing(model: str, backend: str | None) -> str:
  resolved = llm.resolve_backend(backend or "auto", model)
  if resolved == "huggingface" and not os.environ.get(llm.TOKEN_ENV):
    return f"${llm.TOKEN_ENV} is not set and {model!r} routes to the HF router"
  if resolved == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
    return f"$ANTHROPIC_API_KEY is not set and {model!r} routes to the SDK"
  return ""


def fly(configs: list[dict], results: Path, parallel: int,
        wall_limit_s: float, python: str = sys.executable) -> list[Path]:
  """Spawn the runs, at most `parallel` at once; return the record paths."""
  results.mkdir(parents=True, exist_ok=True)
  scratch = Path(tempfile.mkdtemp(prefix="pluggy-experiment-"))
  pending = list(configs)
  live: list[tuple[dict, subprocess.Popen, float, Path, Path]] = []
  written: list[Path] = []
  env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
  while pending or live:
    while pending and len(live) < parallel:
      cfg = pending.pop(0)
      rid = cfg["runId"]
      cfg_path = scratch / f"{rid}.config.json"
      cfg_path.write_text(json.dumps(cfg))
      out = results / f"{rid}.json"
      partial = scratch / f"{rid}.partial.jsonl"
      proc = subprocess.Popen(
        [python, "-m", "pluggybot.evaluation.run", str(cfg_path), str(out),
         str(partial)], cwd=REPO, env=env,
        stdout=open(scratch / f"{rid}.log", "w"), stderr=subprocess.STDOUT)
      live.append((cfg, proc, time.time(), out, partial))
      print(f"started {rid} (pid {proc.pid})", flush=True)
    time.sleep(1.0)
    still = []
    for cfg, proc, t0, out, partial in live:
      code = proc.poll()
      wall = time.time() - t0
      if code is None and wall < wall_limit_s:
        still.append((cfg, proc, t0, out, partial))
        continue
      if code is None:
        # PAST THE LIMIT: kill it and write what it left. `end: "killed"`,
        # never a completed day, never a death of either kind.
        proc.kill()
        proc.wait()
        events = load_events(partial) if partial.exists() else []
        record = validate(build_record(
          cfg, None, events, wall,
          datetime.fromisoformat(cfg["startedAt"]), hashes=cfg["dataHashes"]))
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"KILLED {cfg['runId']} after {wall:.0f} s wall at "
              f"t={record['simSeconds']:.0f}", flush=True)
      elif code != 0 or not out.exists():
        log = (scratch / f"{cfg['runId']}.log").read_text()[-1500:]
        print(f"FAILED {cfg['runId']} (exit {code}):\n{log}", flush=True)
        continue
      else:
        rec = json.loads(out.read_text())
        print(f"done {rec['runId']}: {rec['end']} at t={rec['simSeconds']:.0f}"
              f", {rec['mind']['decisions']} decisions, voluntary "
              f"{rec['charging']['voluntary']['chosen']}, fallbacks "
              f"{rec['mind']['fallbacks']}", flush=True)
      written.append(out)
    live = still
  return written


def summarise(rollup_path: Path) -> None:
  doc = json.loads(rollup_path.read_text())
  for s in doc["series"]:
    ch, m = s["charging"], s["mind"]
    print(f"\n{s['world']} / {s['arm']} / {s['pack']} / {s['model']}"
          + (f" / {s['label']}" if s.get("label") else "")
          + f": n={s['n']}, ends {s['ends']}, current data files: "
          f"{s['current']}")
    print(f"  deadline {s['deadlineS']} s, {s['parallel']} sim(s) sharing "
          f"the box" + (f", rung {s['rung']}" if s.get("rung") else "")
          + (f", origin {s['origin']}" if s.get("origin") else ""))
    # THE STATIC MAP REPORT (issue #127), printed before anything flown --
    # it is the cheapest thing in this summary and the only part of it that
    # cost no sim-hours to produce.
    emap = s["mind"].get("eventMap")
    if emap:
      print(f"  event map: {emap['n']} run(s), rows {emap['rows']['values']}, "
            f"edits {emap['edits']['values']}; wrote a charging rule "
            f"{emap['charges']}/{emap['n']}, kept an `ask` row "
            f"{emap['keepsAsk']}/{emap['n']}, mapped `decision_failed` "
            f"{emap['mapsFailure']}/{emap['n']}; thresholds "
            f"{emap['ordered']}")
      print(f"  rows fired {emap['fired']}, actions failed {emap['failed']}")
    print(f"  voluntary charges chosen {ch['voluntaryChosen']['values']}, "
          f"honoured {ch['voluntaryHonoured']['values']}; deferred "
          f"{ch['deferred']['values']}; forced {ch['forced']['values']}")
    print(f"  decisions {m['decisions']['values']}, fallback rate "
          f"{m['fallbackRate']['values']} ({m['fallbackReasons']}), "
          f"wall/decision median {m['wallS']['median']} s")
    # ...and the same rate split by class, because only one half is judged
    # (issue #141). Printed even when the arm has no limit: "8 fallbacks,
    # all of them the policy" is the sentence that stops a reader concluding
    # the box ate the day.
    print(f"  of which failure {m['fallbackClasses']['failure']} "
          f"{m['fallbackFailureRate']['values']}, policy "
          f"{m['fallbackClasses']['policy']} "
          f"{m['fallbackPolicyRate']['values']}")
    print(f"  deaths {s['survival']['deaths']}, killed {s['killed']}, "
          f"min pack {s['survival']['minFraction']['values']}")
    # The disqualifier, out loud: a run silently dropped from the survival
    # column is a run whose absence nobody notices (issue #117).
    if s["survival"]["excluded"]:
      print(f"  survival n={s['survival']['n']}/{s['n']} -- excluded at "
            f"fallbackLimit {s['fallbackLimit']} (failure class):")
      for e in s["survival"]["excluded"]:
        print(f"    {e['runId']}: {e['why']}")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--world", choices=("home", "room_hub"), default="home")
  ap.add_argument("--arm", choices=ARMS, default="guarded")
  ap.add_argument("--pack", choices=("demo", "hosting"), default="hosting")
  ap.add_argument("-n", "--runs", type=int, default=5)
  ap.add_argument("--parallel", type=int, default=1)
  ap.add_argument("--max-sim-time", type=float, default=3600.0)
  ap.add_argument("--model", default=None,
                  help=f"the deciding model (guarded); default {DEFAULT_MODEL}")
  ap.add_argument("--backend", default=None, choices=llm.BACKENDS)
  ap.add_argument("--errand", default="draw")
  ap.add_argument("--no-tasks", action="store_true")
  ap.add_argument("--no-metabolism", action="store_true")
  ap.add_argument("--seed", type=int, default=0, help="first replicate index")
  ap.add_argument("--origin", default=ORIGIN_DEFAULT, choices=ORIGINS,
                  help="which event map the agent starts with (issue #127): "
                       "`none` is the arm with no map at all, which is how "
                       "A0 and A1 were flown; `seeded` starts it with "
                       "today's loop re-expressed as rows; `unseeded` starts "
                       "it with nothing and tells it to configure itself. "
                       "AN ABLATION, not a rung -- `unseeded` moves the "
                       "prompt as well as the configuration, so a null is "
                       "strong evidence and a difference is weak")
  ap.add_argument("--rung", default="A0", choices=("A0", "A1"),
                  help="which rung of the autonomous ladder (issue #115): "
                       "A0 is the null, A1 adds the survival clock to what "
                       "the model is shown. Ignored on the other arms")
  ap.add_argument("--label", default="",
                  help="what the BOX was, in a word (issue #117): `quiet` "
                       "for a machine with nothing else on it. It joins the "
                       "series key, so a quiet series and a loaded one are "
                       "committed side by side instead of averaged into one")
  ap.add_argument("--results", default=str(RESULTS))
  ap.add_argument("--wall-limit", type=float, default=None,
                  help=f"seconds before a run is killed (default "
                       f"{WALL_PER_SIM_S} x --max-sim-time, at least "
                       f"{WALL_FLOOR_S:.0f})")
  ap.add_argument("--telemetry", action="store_true",
                  help="also write a PluggyWorld recording beside each record")
  ap.add_argument("--rollup", action="store_true",
                  help="only re-aggregate the results directory")
  args = ap.parse_args()
  results = Path(args.results)
  if not args.rollup:
    model = None
    if args.arm != "scripted":
      model = args.model or os.environ.get("PLUGGY_MODEL") or DEFAULT_MODEL
      missing = credentials_missing(model, args.backend)
      if missing:
        print(f"refusing to fly the {args.arm} arm: {missing}", file=sys.stderr)
        return 2
    wall_limit = wall_limit_for(args.max_sim_time, args.wall_limit)
    hashes = data_hashes(args.world)
    configs = []
    for k in range(args.runs):
      started = datetime.now(timezone.utc)
      cfg = {"world": args.world, "arm": args.arm, "pack": args.pack,
             "model": model, "backend": args.backend, "seed": args.seed + k,
             "errand": args.errand, "tasks": not args.no_tasks,
             "metabolism": not args.no_metabolism,
             "label": args.label, "rung": args.rung,
             "origin": origin_for(args.arm, args.origin),
             "maxSimS": args.max_sim_time, "freshState": True,
             "parallel": args.parallel, "wallLimitS": wall_limit,
             "startedAt": started.isoformat(), "dataHashes": hashes}
      cfg["runId"] = run_id(cfg, started)
      if args.telemetry:
        cfg["telemetry"] = str(results / f"{cfg['runId']}.telemetry.jsonl.gz")
      configs.append(cfg)
      time.sleep(1.1)                 # distinct second-resolution run ids
    fly(configs, results, max(1, args.parallel), wall_limit)
  path = write_rollup(results)
  print(f"\nrollup -> {path}")
  summarise(path)
  return 0


if __name__ == "__main__":
  sys.exit(main())
