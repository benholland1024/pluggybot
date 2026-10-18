"""Graded practice on the humanoid (issue #493, stages 2 and 3): the robot's
mind writes a procedure for the chest's lid (`mind/practice.py`), the
humanoid tries it from the counter's set-out, the world grades the try
through stage 1's door, and the mind is shown how it went and writes again
(`evaluation/practice.py`) -- up to `--attempts` tries a chest, a fresh
robot a chest, a pass admitted to its library.

  --chests A-B|A,B,..  the chests (default 0-31: the public set-outs)
  --held-out N         N chests nobody has flown instead, by the frozen
                       build's seed (`attempts.seed_of`; a dirty tree is
                       refused), or `--seed`
  --attempts N         tries a chest (default 5: the curve's S@1..S@5)
  --pull               stage 3's vocabulary, the compliant pull beside
                       stage 2's; without it, baseline (a'): position only
  --model ID           the mind (default the deployed routine model,
                       `$PLUGGY_MODEL`'s); its key is $HF_TOKEN
  --jobs J             chests at once, each in a process of its own: a
                       chest spends most of its wall time waiting on the
                       mind, so J may pass the cores, memory allowing
  --into DIR           the record: rows.jsonl, a row an attempt, written
                       as each chest ends
  --no-picture         the first turn without the head camera's picture
  --read DIR           a kept record, read as the per-task record (its
                       curve, each regime apart) with what it cost

Usage:
  systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \\
    uv run python scripts/practice.py --chests 0-31 --jobs 6 --into DIR
  uv run python scripts/practice.py --held-out 32 --pull --jobs 6 --into DIR
  uv run python scripts/practice.py --read DIR
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attempts as att  # noqa: E402

from pluggybot.evaluation import attempts as at  # noqa: E402

#: The deployed routine model (#225; call 4: it writes first).
MODEL = "zai-org/GLM-5.3-Flash:cheapest"
#: The mind's answers stream (`mind.llm.stream_fetch`), read this long, s.
READ_S = 180.0


def fly(job) -> list[dict]:
  """One chest's practice, a fresh robot: its rows, each with what it cost.
  A chest whose practice raised is a row saying so (`error`)."""
  k, attempts, pull, model, picture, info, price = job
  from pluggybot.evaluation import practice as ep
  from pluggybot.mind.llm import HFClient, stream_fetch
  from pluggybot.mind.practice import Practice
  from pluggybot.procedure.library import Library
  from pluggybot.procedure.steps import hand_facts
  facts = hand_facts(pull=pull)
  mind = Practice(HFClient(timeout=READ_S, fetch=stream_fetch), model, facts,
                  ep.job_for(at.Task(chest=k, body="humanoid", mount="counter")))
  skill = f"{model} {'with' if pull else 'without'} the pull"
  try:
    rows = ep.practise(k, mind, attempts, library=Library(facts), picture=picture,
                       info={**info, "skill": skill})
  except Exception as e:
    traceback.print_exc()
    task = at.Task(chest=k, body="humanoid", mount="counter")
    return [{"task": task.as_dict(), "attempt": len(mind.turns), "heldOut": task.held_out,
             "process": "practice", "skill": skill, "error": f"{type(e).__name__}: {e}",
             "run": {k_: info[k_] for k_ in ("commit", "dirty", "seed") if k_ in info}}]
  for r in rows:
    r["run"] |= {"model": model, "pull": pull, "picture": picture}
    p = r["practice"]
    r["usd"] = (None if price is None
                else round((p["tokensIn"] * price[0] + p["tokensOut"] * price[1]) / 1e6, 5))
  return rows


def show(row: dict) -> str:
  if "error" in row:
    return f"  chest {row['task']['chest']}: ERROR {row['error']}"
  p = row["practice"]
  return (att.show(row) + f" [{row.get('steps')} verbs, {p['tokensIn']}+{p['tokensOut']} tokens"
          + (f", ${row['usd']:.4f}" if row.get("usd") is not None else "") + "]")


def read(into: str) -> None:
  """The record as the per-task record (`attempts.py --read`), and what the
  mind cost: tokens, dollars and wall time, each regime apart."""
  att.read(into)
  rows = [json.loads(line) for line in open(Path(into) / "rows.jsonl")]
  regimes: dict = {}
  for r in rows:
    if "practice" not in r:
      continue
    key = (r["run"].get("model"), r["run"].get("pull"))
    regimes.setdefault(key, []).append(r)
  for (model, pull), mine in regimes.items():
    tin = sum(r["practice"]["tokensIn"] for r in mine)
    tout = sum(r["practice"]["tokensOut"] for r in mine)
    usd = [r.get("usd") for r in mine]
    calls = sum(r["practice"]["calls"] for r in mine)
    print(f"THE MIND, {model}, {'with' if pull else 'without'} the pull: {len(mine)} attempts, "
          f"{calls} answers, {tin} tokens in and {tout} out"
          + (f", ${sum(usd):.2f}" if None not in usd else ", cost unknown")
          + f", {sum(r['practice']['llmWallS'] for r in mine) / 3600:.1f} h waiting on it")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--chests", default="0-31")
  ap.add_argument("--held-out", type=int, default=None, metavar="N")
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--attempts", type=int, default=5)
  ap.add_argument("--pull", action="store_true")
  ap.add_argument("--model", default=os.environ.get("PLUGGY_MODEL", MODEL))
  ap.add_argument("--jobs", type=int, default=1)
  ap.add_argument("--into", default=None)
  ap.add_argument("--no-picture", action="store_true")
  ap.add_argument("--read", default=None, metavar="DIR")
  args = ap.parse_args(argv)
  if args.read:
    read(args.read)
    return
  info = {"process": "practice"} | att.build()
  if args.held_out is not None:
    if info["dirty"] and args.seed is None:
      ap.error("a held-out draw is a frozen build's: commit the tree first (or name --seed)")
    seed = args.seed if args.seed is not None else at.seed_of(info["commit"])
    ks = at.held_out(seed, args.held_out)
    info["seed"] = seed
    print(f"held out: {args.held_out} chests by seed {seed}: {ks}")
  else:
    ks = att.chests(args.chests)
  from pluggybot.mind.llm import HFClient
  price = HFClient(timeout=30.0).pricing(args.model)
  print(f"the mind: {args.model}, ${price[0]}/Mtok in, ${price[1]}/Mtok out" if price
        else f"the mind: {args.model}, its price unknown")
  out = None
  if args.into:
    Path(args.into).mkdir(parents=True, exist_ok=True)
    out = Path(args.into) / "rows.jsonl"
  jobs = [(k, args.attempts, args.pull, args.model, not args.no_picture, info, price) for k in ks]
  with get_context("spawn").Pool(max(1, args.jobs), maxtasksperchild=1) as pool:
    for rows in pool.imap_unordered(fly, jobs):
      for r in rows:
        print(show(r), flush=True)
        if out is not None:
          with open(out, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
  if args.into:
    read(args.into)


if __name__ == "__main__":
  main()
