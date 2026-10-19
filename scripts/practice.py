"""Graded practice on the humanoid (issue #493, stages 2 and 3): the robot's
mind writes a procedure for the chest's lid (`mind/practice.py`), the
humanoid tries it from the counter's set-out, the world grades the try
through stage 1's door, and the mind is shown how it went and writes again
(`evaluation/practice.py`) -- up to `--attempts` tries a chest, a pass
admitted to its library.

Two protocols (Evaluation.md §4):
  FRESH, the default: a fresh robot a chest, nothing carried.
  KEPT, `--acquire`: one robot practices on public chests in turn, carrying
  only what passed (`evaluation.practice.keep`), until it passes `--until`
  chests running at their first try or has practiced `--max-chests`; then
  each held-out chest gets a copy of it, frozen, and the bar is read there.

  --chests A-B|A,B,..  the chests (default 0-31: the public set-outs)
  --held-out N         N chests nobody has flown instead, by the frozen
                       build's seed (`attempts.seed_of`; a dirty tree is
                       refused), or `--seed`
  --acquire A-B|A,B,.. KEPT: first practice on these public chests, in turn
  --until N            ...until N chests running pass at their first try (3)
  --max-chests M       ...or M are practiced (16)
  --give-up-after G    ...and nothing more is flown where nothing is kept
                       after G chests (6)
  --kept FILE          KEPT: the held-out chests flown with what a record
                       kept (its kept.json), no practice first
  --notes              KEPT: the robot also carries notes and a draft of its
                       own from chest to chest (`Practice.carry`), written as
                       each practice chest ends, pass or fail
  --measure            the depth camera's `measure` beside the hands' verbs:
                       sight beside feel, an arm of its own
  --verb-tests         a verb may be an `if`'s test, its failing read rather
                       than ending the procedure (`WorldFacts.verb_tests`)
  --path               a hand's path (`via`, `follow`: one motion from one
                       posture) and sin, cos, atan2, sqrt (`WorldFacts.math`)
  --practice-attempts N  KEPT: tries a PRACTICE chest (default --attempts);
                       the held-out chests keep --attempts
  --early-stop         the chests stop once the bar is out of reach
                       (`evaluation.practice.out_of_reach`; Ben, 2026-10-18)
  --attempts N         tries a chest (default 5: the curve's S@1..S@5)
  --pull               stage 3's vocabulary, the compliant pull beside
                       stage 2's; without it, baseline (a'): position only
  --model ID           the mind (default the deployed routine model,
                       `$PLUGGY_MODEL`'s); its key is $HF_TOKEN
  --jobs J             chests at once, each in a process of its own: a
                       chest spends most of its wall time waiting on the
                       mind, so J may pass the cores, memory allowing
  --into DIR           the record: rows.jsonl, a row an attempt, written
                       as each chest ends; KEPT's kept.json, and
                       stopped.json where the chests stopped early
  --no-picture         the first turn without the head camera's picture
  --read DIR           a kept record, read as the per-task record (its
                       curve, each regime apart) with what it cost

Usage:
  systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \\
    uv run python scripts/practice.py --chests 0-31 --jobs 6 --into DIR
  uv run python scripts/practice.py --held-out 32 --pull --jobs 6 --into DIR
  uv run python scripts/practice.py --acquire 0-31 --held-out 32 --pull \\
    --early-stop --model zai-org/GLM-5.2:cheapest --jobs 6 --into DIR
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
from pluggybot.evaluation import practice as ep  # noqa: E402

#: The deployed routine model (#225; call 4: it writes first).
MODEL = "zai-org/GLM-5.3-Flash:cheapest"
#: The mind's answers stream (`mind.llm.stream_fetch`), read this long, s.
READ_S = 180.0


def fly(job: dict) -> list[dict]:
  """One chest's practice: its rows, each with what it cost. The robot
  carries `job["kept"]` (none in FRESH). A chest whose practice raised is a
  row saying so (`error`)."""
  from pluggybot.mind.llm import HFClient, stream_fetch
  from pluggybot.mind.practice import Practice
  from pluggybot.procedure.library import Library
  from pluggybot.procedure.steps import hand_facts
  k, pull, model, info, price = (job[n] for n in ("chest", "pull", "model", "info", "price"))
  facts = hand_facts(pull=pull, measure=job["measure"], verb_tests=job["verb_tests"],
                     path=job["path"])
  task = at.Task(chest=k, body="humanoid", mount="counter")
  mind = Practice(HFClient(timeout=READ_S, fetch=stream_fetch), model, facts, ep.job_for(task))
  skill = (f"{model} {'with' if pull else 'without'} the pull"
           + (", notes" if job["notes"] else "") + (", measure" if job["measure"] else "")
           + (", verb tests" if job["verb_tests"] else "") + (", path" if job["path"] else ""))
  info = {**info, "skill": skill, "process": job["process"]}
  try:
    rows = ep.practice_on(k, mind, job["attempts"], library=Library(facts), kept=job["kept"],
                          carried=job["carried"], write_notes=job["write_notes"],
                          picture=job["picture"], info=info)
  except Exception as e:
    traceback.print_exc()
    return [{"task": task.as_dict(), "attempt": len(mind.turns), "heldOut": task.held_out,
             "process": job["process"], "skill": skill, "error": f"{type(e).__name__}: {e}",
             "run": {k_: info[k_] for k_ in ("commit", "dirty", "seed") if k_ in info}}]
  for r in rows:
    r["run"] |= {"model": model, "pull": pull, "picture": job["picture"],
                 "notes": job["notes"], "measure": job["measure"],
                 "verbTests": job["verb_tests"], "path": job["path"],
                 "attempts": job["attempts"]}
    p = r["practice"]
    r["usd"] = (None if price is None
                else round((p["tokensIn"] * price[0] + p["tokensOut"] * price[1]) / 1e6, 5))
  return rows


def show(row: dict) -> str:
  if "error" in row:
    return f"  chest {row['task']['chest']}: ERROR {row['error']}"
  p = row["practice"]
  return (att.show(row) + f" [{row.get('steps')} verbs, {p['tokensIn']}+{p['tokensOut']} tokens"
          + (f", ${row['usd']:.4f}" if row.get("usd") is not None else "") + "]"
          + (f" ADMITTED {row['admitted']}" if row.get("admitted") else ""))


def read(into: str) -> None:
  """The record as the per-task record (`attempts.py --read`), what KEPT's
  practice kept and where the chests stopped, and what the mind cost:
  tokens, dollars and wall time, each regime apart."""
  att.read(into)
  rows = [json.loads(line) for line in open(Path(into) / "rows.jsonl")]
  regimes: dict = {}
  for r in rows:
    if "practice" not in r:
      continue
    key = (r["run"].get("model"), r["run"].get("pull"), r.get("process"))
    regimes.setdefault(key, []).append(r)
  for (model, pull, process), mine in regimes.items():
    tin = sum(r["practice"]["tokensIn"] for r in mine)
    tout = sum(r["practice"]["tokensOut"] for r in mine)
    usd = [r.get("usd") for r in mine]
    calls = sum(r["practice"]["calls"] for r in mine)
    print(f"THE MIND, {process}, {model}, {'with' if pull else 'without'} the pull: "
          f"{len(mine)} attempts, {calls} answers, {tin} tokens in and {tout} out"
          + (f", ${sum(usd):.2f}" if None not in usd else ", cost unknown")
          + f", {sum(r['practice']['llmWallS'] for r in mine) / 3600:.1f} h waiting on it")
  kept_file = Path(into) / "kept.json"
  if kept_file.exists():
    k = json.loads(kept_file.read_text())
    print(f"THE PRACTICE FIRST: {k['practice']['ended']}; each chest's first pass, in turn: "
          + ", ".join(f"{c}: {f}" for c, f in k["practice"]["firsts"]))
    for e in k["kept"]:
      print(f"  kept {e['name']}, passed chest {e['chest']} at its try {e['attempt']}")
    carried = k["practice"].get("carried") or {}
    if carried.get("notes") or carried.get("draft"):
      print(f"  its notes at the end ({len(carried.get('notes', ''))} characters): "
            + carried.get("notes", "").replace("\n", " ")[:400])
      print(f"  its draft: {len(carried.get('draft', ''))} characters")
  stopped = Path(into) / "stopped.json"
  if stopped.exists():
    s = json.loads(stopped.read_text())
    print(f"STOPPED EARLY after {s['after']} of {s['of']} chests: {s['failedWithin']} failed "
          f"within {s['within']}, so {s['bar']:.0%} within {s['within']} is out of reach")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--chests", default="0-31")
  ap.add_argument("--held-out", type=int, default=None, metavar="N")
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--acquire", default=None, metavar="CHESTS")
  ap.add_argument("--until", type=int, default=ep.UNTIL)
  ap.add_argument("--max-chests", type=int, default=ep.MAX_CHESTS)
  ap.add_argument("--give-up-after", type=int, default=ep.GIVE_UP_AFTER)
  ap.add_argument("--kept", default=None, metavar="FILE")
  ap.add_argument("--early-stop", action="store_true")
  ap.add_argument("--notes", action="store_true")
  ap.add_argument("--measure", action="store_true")
  ap.add_argument("--verb-tests", action="store_true")
  ap.add_argument("--path", action="store_true")
  ap.add_argument("--practice-attempts", type=int, default=None, metavar="N")
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

  def write(rows):
    for r in rows:
      print(show(r), flush=True)
      if out is not None:
        with open(out, "a") as f:
          f.write(json.dumps(r, default=str) + "\n")

  def job(k, kept, process, carried=None, write_notes=False, attempts=None):
    return {"chest": k, "attempts": attempts or args.attempts, "pull": args.pull,
            "model": args.model, "picture": not args.no_picture, "info": dict(info),
            "price": price, "kept": kept, "process": process, "carried": carried,
            "write_notes": write_notes, "notes": args.notes, "measure": args.measure,
            "verb_tests": args.verb_tests, "path": args.path}

  kept, process, carried = [], "practice", None
  with get_context("spawn").Pool(max(1, args.jobs), maxtasksperchild=1) as pool:
    if args.acquire or args.kept:
      process = "apply"
      if args.kept:
        was = json.loads(Path(args.kept).read_text())
        kept = was["kept"]
        carried = (was.get("practice") or {}).get("carried") if args.notes else None
      else:
        public = {k: v for k, v in info.items() if k != "seed"}

        def run_chest(k, kept_, carried_):
          return pool.apply(fly, (job(k, kept_, "acquire", carried_ if args.notes else None,
                                      write_notes=args.notes,
                                      attempts=args.practice_attempts) | {"info": public},))
        kept, how = ep.acquire(run_chest, att.chests(args.acquire), args.until,
                               args.max_chests, args.give_up_after, write)
        carried = how["carried"] if args.notes else None
        if args.into:
          (Path(args.into) / "kept.json").write_text(
            json.dumps({"kept": kept, "practice": how}, indent=1))
        if not kept:
          print(f"KEPT: {how['ended']}; no chest flown with nothing kept", flush=True)
          pool.terminate()
          if args.into:
            read(args.into)
          return
    stopped = ep.flown(pool.imap_unordered(fly, [job(k, kept, process, carried) for k in ks]),
                       len(ks), args.early_stop, write)
    if stopped is not None:
      print(f"STOPPED EARLY: {stopped}", flush=True)
      pool.terminate()
      if args.into:
        (Path(args.into) / "stopped.json").write_text(json.dumps(stopped, indent=1))
  if args.into:
    read(args.into)


if __name__ == "__main__":
  main()
