"""The robot's model of the chest, graded (issue #466's stage 3; SimNotes,
"The robot's model, graded").

Over the probes `scripts/probe_chest.py --into DIR` kept (each set-out's
record, its picture and OUR truth beside it), one or more CONDITIONS, each
graded by us against the world's chest and the best-expressible reference
(`evaluation.model`), never pooled:

  model      the robot's own: an author's structure (`imagination.author`,
             the deployed model by default) placed where depth put the box,
             its unknowns fitted to the record, judged and sent back where
             it left too much (`imagination.model`, up to `MAX_ROUNDS`
             rounds); each row carries its `settings`, and the summary
             never pools rows whose settings differ (#481)
  reference  the fitter alone: the reference's own structure and geometry,
             its lid's dynamics left unknown (`evaluation.model.
             reference_template`)
  leak       the same with its hinge moved (`--offsets`): what a geometry
             error leaks into the dynamics (#466's decision 2)

  --probes DIR     the kept probes (probe_K.npz, truth_K.npz, rows.jsonl)
  --n N, --from K  set-outs K..K+N-1 (default all kept); --ks K,... by number
  --into DIR       each set-out's result (k_<cond>.json) and rows.jsonl
  --conditions     a comma list (default model)
  --workers W      rollout workers a set-out (default 5), here or on pods
                   (--remote HOST:PORT,...: the worker over ssh, the same
                   frames; each pod has the commit installed, `training/pod.sh
                   setup-sim`), --slots S set-outs at once on each (default 2;
                   here, S at once)
  --author MODEL   the author's model id ($PLUGGY_MODEL, else the deployed
                   pick); its key is $HF_TOKEN
  --summary        the tables off --into's rows, nothing fitted: per settings,
                   the parameters, the author's structures and its rate --
                   passes and the lid's hinge by round, with intervals, false
                   passes, the answer budget, tokens, dollars and wall time;
                   --also DIR,... more batches of the same set-outs, read as
                   more imaginings of each, in that order, and the first of
                   them to pass
  --demo K --record PATH   set-out K flown in the house with the telemetry
                   recorder on: the probe, the robot lying still while it
                   imagines (its workers here or --remote), and the
                   `imagined` event the site draws its ghost from; the scene
                   of the demo's world (`DEMO_WORLD`) and its textures beside
                   the recording. `--attempts N` imagines afresh N times at
                   once and keeps the first, in order, whose model passed
                   its bars -- a demo's, which the event says (`attempts`);
                   the batch measures one

Usage:
  uv run python scripts/imagine_chest.py --probes DIR --into OUT --n 4
  uv run python scripts/imagine_chest.py --probes DIR --into OUT --remote H:P,H2:P2 \\
    --workers 5 --slots 2 --conditions model,reference
  uv run python scripts/imagine_chest.py --into OUT --probes DIR --summary
  MUJOCO_GL=egl uv run python scripts/imagine_chest.py --demo 0 --record demo.jsonl.gz
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from pluggybot.evaluation import imagined as im  # noqa: E402
from pluggybot.evaluation import model as em  # noqa: E402
from pluggybot.evaluation import probe as ep  # noqa: E402
from pluggybot.imagination import fit as ft  # noqa: E402
from pluggybot.imagination import model as mm  # noqa: E402
from pluggybot.imagination.worker import Imaginations, unpack  # noqa: E402
from pluggybot.legs import probe as pr  # noqa: E402

#: The deployed author (CLAUDE.md, "Which model is $PLUGGY_MODEL").
DEPLOYED = "zai-org/GLM-5.3-Flash:cheapest"
#: What the author is told beyond #480's batch (#481's stage 1): what the
#: jaws felt in its first turn, and which end of a range a fitted value
#: stopped at. A row carries it in its settings.
LEVERS = ("felt first", "which end")
#: #480's rows carry no settings: they were these.
STAGE3 = {"maxRounds": 3, "maxTokens": 32000, "levers": []}
#: The bar to go on (#481, Ben, 2026-10-08): this share of imaginings pass
#: their bars by the last round.
BAR = 0.4
#: An answer past this many tokens came only with #481's larger budget.
OLD_TOKENS = 32000
#: The hinge's moves the leak flies, mm (along the box into it, up): what
#: depth leaves along it (stage 2: the planned hinge within 1.8 mm), and the
#: heights a lid's thickness could leave (stage 2's foresight: 12.5 mm).
OFFSETS = ((2.0, 0.0), (-2.0, 0.0), (0.0, 5.0), (0.0, 12.5))
KEY = os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
#: The author's answers stream (`mind.llm.stream_fetch`: the router's gateway
#: answers 504 to a request silent 120 s), and each read may wait this long, s.
READ_S = 180.0
#: The demo's world on the wire: the house with the quadruped and one drawn
#: chest, a world of its own (a replayer picks its scene off `model`).
DEMO_WORLD = "home_quad_chest"
#: The demo lies still this long while it imagines, and this long with its
#: model made, sim s.
THINK_S, AFTER_S = 6.0, 8.0


def _ssh(where: str, slot: int) -> tuple[list[str], str]:
  host, port = where.rsplit(":", 1)
  # ⚠ Ten minutes unanswered before a slot is gone: at three keepalives the
  # connections to three pods in three places ended within minutes of each
  # other, and took eleven set-outs and the demo with them.
  return (["ssh", "-i", KEY, "-p", port, "-o", "StrictHostKeyChecking=accept-new",
           "-o", f"ControlPath=/tmp/pb466-{os.getpid()}-{host}-{port}-{slot}",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=20"], f"root@{host}")


def open_master(where: str, slot: int) -> None:
  """A slot's one connection to its pod, which its workers' channels share:
  opened first, or workers starting at once each try to be it. ⚠ A slot a
  connection, and a run its own (the path names its process): sshd refuses
  a connection's eleventh channel (MaxSessions). Closed when the run ends,
  or it outlives the run by its persistence."""
  import atexit
  import subprocess
  opts, dest = _ssh(where, slot)
  subprocess.run(opts + ["-o", "ControlMaster=yes", "-o", "ControlPersist=7200", "-fN", dest],
                 check=True)
  atexit.register(subprocess.run, opts + ["-O", "exit", dest], capture_output=True)


def remote_argv(where: str, slot: int = 0) -> list[str]:
  """The worker on a pod, over ssh: its stdin and stdout are the frames."""
  opts, dest = _ssh(where, slot)
  return [*opts, "-o", "ControlMaster=no", dest,
          "cd /root/pluggybot && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
          "MKL_NUM_THREADS=1 .venv/bin/python -m pluggybot.imagination.worker --seed 0"]


def load(probes: Path, k: int):
  out = pr.Probed.from_wire(*unpack((probes / f"probe_{k}.npz").read_bytes()))
  truth = np.load(probes / f"truth_{k}.npz")
  return out, truth


def setting_of(k: int, out: pr.Probed, truth) -> tuple[im.Setting, float]:
  """Where the world's chest stands in the robot's map at the record's start
  (the truth through the true pose and the belief), and where its handle
  hung on its pin then."""
  so = ep.set_out(k)
  chest_map = ep.in_map(so.chest, tuple(truth["start_true"]), out.record.start.pose)
  return (im.Setting(chest_x=chest_map[0], chest_y=chest_map[1], chest_yaw=chest_map[2]),
          float(truth["rows"][0, 1]))


def _clean(v):
  """JSON-ready: numpy numbers as floats, tuples as lists."""
  if isinstance(v, dict):
    return {str(k): _clean(x) for k, x in v.items()}
  if isinstance(v, (list, tuple)):
    return [_clean(x) for x in v]
  if isinstance(v, (np.floating, np.integer)):
    return v.item()
  if isinstance(v, np.ndarray):
    return v.tolist()
  return v


def graded(document, k, out, truth) -> dict:
  setting, pin = setting_of(k, out, truth)
  g = em.grade(document, ep.set_out(k).lid, setting, pin)
  return {"graded": g, "errors": em.errors(g)}


def run_reference(k, out, truth, pool, offset=(0.0, 0.0)) -> dict:
  setting, pin = setting_of(k, out, truth)
  t = em.reference_template(ep.set_out(k).lid, setting, pin, hinge_off_mm=offset)
  t0 = time.time()
  fitted = ft.fit(t, out.record, pr.fit_rows(out), pool)
  judged = mm.judge(out.record, fitted.readings, pr.phases_of(out))
  return {"fitted": fitted.as_dict(), "wallS": round(time.time() - t0, 1),
          "judged": [{k_: v for k_, v in p.items() if k_ != "trace"} for p in judged],
          "offsetMm": list(offset), **graded(fitted.document, k, out, truth)}


def settings(author_model: str) -> dict:
  """What a model's row was produced by (#481): its author, its rounds, its
  answer budget and what the author was told beyond #480's batch."""
  from pluggybot.imagination import author as au
  return {"author": author_model, "maxRounds": mm.MAX_ROUNDS, "maxTokens": au.MAX_TOKENS,
          "levers": list(LEVERS)}


def run_model(k, out, truth, pool, author_model, price=None) -> dict:
  from pluggybot.imagination.author import Author
  from pluggybot.mind.llm import HFClient, stream_fetch
  s = pr.sizes(out.record, out.guess)
  if isinstance(s, str):
    return {"error": f"no sizes: {s}"}
  author = Author(HFClient(timeout=READ_S, fetch=stream_fetch), author_model)
  t0 = time.time()
  from pluggybot.legs.imagined import felt
  m = mm.imagine(author, out.record, pr.phases_of(out), pr.fit_rows(out), s.as_dict(),
                 s.origin, s.yaw, pr.did(out, s),
                 pr.exposed(out.picture) if out.picture else None, pool,
                 felt=felt(out.record, out.friction))
  row = {"model": m.as_dict(), "sizes": s.as_dict(), "wallS": round(time.time() - t0, 1),
         "author": author_model, "picture": out.picture is not None,
         "settings": settings(author_model),
         "priceUsdPerMtok": None if price is None else list(price)}
  lid = ep.set_out(k).lid
  row["rounds"] = []
  for r in m.rounds:
    if r.fitted is None:
      row["rounds"].append(None)
      continue
    g = graded(r.fitted.document, k, out, truth)
    jid = (g["graded"].get("model") or {}).get("joint")
    g["ranges"] = em.ranges_graded(r.turn.template.raw, jid, g["graded"].get("lidPart"),
                                   lid) if r.turn.template is not None else None
    row["rounds"].append(g)
  return row


def one(k, args, argv) -> dict:
  probes, into = Path(args.probes), Path(args.into)
  out, truth = load(probes, k)
  base = {"k": k, "catchN": round(ep.set_out(k).lid.catch_n, 2), "spring": k % 2 == 1}
  if not out.ok or out.record is None:
    return base | {"skipped": out.why}
  rows = []
  with Imaginations(args.workers, argv=argv) as pool:
    for cond in args.conditions:
      dest = into / f"{k}_{cond}.json"
      if dest.exists():
        kept = json.loads(dest.read_text())
        if "error" not in kept:                     # a rerun redoes what failed
          rows.append(kept)
          continue
      try:
        if cond == "reference":
          got = run_reference(k, out, truth, pool)
        elif cond == "leak":
          got = {"offsets": [run_reference(k, out, truth, pool, off) for off in OFFSETS]}
        else:
          got = run_model(k, out, truth, pool, args.author, getattr(args, "price", None))
      except Exception as e:                                  # noqa: BLE001 -- a row says
        traceback.print_exc()
        got = {"error": f"{type(e).__name__}: {e}"}
      got = _clean(base | {"condition": cond} | got)
      dest.write_text(json.dumps(got))
      rows.append(got)
  return rows


# ---- the tables -----------------------------------------------------------------------

def _get(d, *path):
  """`d` down `path`, None where any step is missing: a round past a
  conversation's end is absent, never an error."""
  for p in path:
    if isinstance(d, list) and isinstance(p, int):
      d = d[p] if 0 <= p < len(d) else None
    else:
      d = d.get(p) if isinstance(d, dict) else None
    if d is None:
      return None
  return d


#: The parameters graded, as the tables name them: (name, key in a grade,
#: index into a list or None, its unit).
PARAMETERS = (("static at 11 deg", "static", 0, "N*m"), ("static at 34 deg", "static", 1, "N*m"),
              ("static at 57 deg", "static", 2, "N*m"), ("first moment", "moment", None, "kg*m"),
              ("friction", "friction", None, "N*m"), ("damping", "damping", None, "N*m*s/rad"),
              ("catch's release", "release", None, "N"), ("catch's hold", "holdNm", None, "N*m"),
              ("range's end", "rangeDeg", None, "deg"),
              ("mass (assumed)", "mass", None, "kg"), ("second moment (assumed)", "second", None,
                                                      "kg*m^2"))
GEOMETRY = (("hinge's axis", "axisDeg", "deg"), ("hinge along the box", "hingeAlongMm", "mm"),
            ("hinge up", "hingeUpMm", "mm"), ("lid's middle", "lidMiddleMm", "mm"))
PHASE_NAMES = ("take", "up0", "down0", "up1", "down1", "sweeps")


def _iv(values, fmt: str = ".4f") -> str:
  iv = em.median_interval(values)
  if iv is None:
    return "--"
  return (f"{iv['median']:{fmt}} [{iv['lo']:{fmt}}, {iv['hi']:{fmt}}], 9 in 10 under "
          f"{iv['p90']:{fmt}}, n {iv['n']}")


def _err(g, side: str, key: str, i):
  v = _get(g, "errors", side, key)
  if v is not None and i is not None:
    v = v[i]
  return None if v is None else abs(v)


def _graded(rows: list, cond: str) -> list[tuple[dict, dict]]:
  """(row, its grade): a model's kept round -- its grade, and the phases
  its fit was judged on, which the model's own rounds keep -- or the
  reference's fit."""
  out = []
  for r in rows:
    if r.get("condition") != cond or "error" in r:
      continue
    if cond == "model":
      kept = _get(r, "model", "kept")
      if kept is not None:
        out.append((r, {**r["rounds"][kept],
                        "judged": _get(r, "model", "rounds", kept, "judged") or []}))
    else:
      out.append((r, r))
  return out


def _lid_right(r: dict, i: int) -> bool:
  """Round `i`'s graded model has a hinge on the lid's line: within 20 mm
  of it and 5 deg (absent is wrong; a perfect 0 is not absent)."""
  m = _get(r, "rounds", i, "graded", "model")
  if not m or m.get("type") != "hinge":
    return False
  got = [m.get("axisDeg"), m.get("hingeAlongMm"), m.get("hingeUpMm")]
  return None not in got and got[0] < 5.0 and abs(got[1]) < 20 and abs(got[2]) < 20


def _has_pin(doc) -> bool:
  """A document whose cube rides a hinge of its own below the lid's."""
  joints = (doc or {}).get("joints") or []
  return sum(1 for j in joints if j.get("type") == "hinge") >= 2


def _tables(title: str, sel: list, replays: dict) -> None:
  """One group's tables: each parameter against the world, the reference,
  and the reference against the world; the geometry; what each phase
  left."""
  print(f"\n== {title}: {len(sel)}")
  for name, key, i, unit in PARAMETERS:
    print(f"  {name} ({unit}), |model - world|: {_iv([_err(g, 'vsWorld', key, i) for _, g in sel])}")
    print(f"      |model - reference|: {_iv([_err(g, 'vsReference', key, i) for _, g in sel])}")
    print(f"      |reference - world|: {_iv([_err(g, 'language', key, i) for _, g in sel])}")
  for name, key, unit in GEOMETRY:
    v = [None if _get(g, "graded", "model", key) is None
         else abs(_get(g, "graded", "model", key)) for _, g in sel]
    print(f"  {name} ({unit}) off the world's: {_iv(v, '.2f')}")
  for ph in PHASE_NAMES:
    fit = [p.get("rmsN") for _, g in sel for p in (g.get("judged") or []) if p["phase"] == ph]
    ref = [(replays.get(r["k"]) or {}).get("reference", {}).get(ph, [None])[0] for r, _ in sel]
    wor = [(replays.get(r["k"]) or {}).get("world", {}).get(ph, [None])[0] for r, _ in sel]
    print(f"  left on {ph} (N): the fit {_iv(fit, '.3f')}")
    print(f"      the reference {_iv(ref, '.3f')}; the world's chest {_iv(wor, '.3f')}")


def _leak_tables(rows: list, flagged: set) -> None:
  """Decision 2's measurement: the reference's fit with its hinge moved,
  beside the same fit unmoved on the same clean set-outs."""
  leak = {r["k"]: r for r in rows if r.get("condition") == "leak" and "error" not in r}
  ref = {r["k"]: r for r in rows if r.get("condition") == "reference" and "error" not in r}
  ks = sorted(k for k in leak if k in ref and k not in flagged)
  if not ks:
    return
  print(f"\n== leak, clean set-outs: {len(ks)}, |fit - world|")
  cols = [("the hinge where it is", [ref[k] for k in ks])]
  for along, up in OFFSETS:
    cols.append((f"the hinge {along:+g} mm along, {up:g} up",
                 [next(o for o in leak[k]["offsets"] if o["offsetMm"] == [along, up]) for k in ks]))
  for label, gs in cols:
    print(f"  {label}:")
    for name, key, i, unit in PARAMETERS[:6]:
      print(f"    {name} ({unit}): {_iv([_err(g, 'vsWorld', key, i) for g in gs])}")
    left = [p.get("rmsN") for g in gs for p in g.get("judged") or [] if p["phase"] == "sweeps"]
    print(f"    left on sweeps (N): {_iv(left, '.3f')}")


def settings_of(r: dict) -> dict:
  """A model row's settings: its own, or #480's batch's, which carried none
  (`STAGE3`)."""
  return r.get("settings") or {"author": r.get("author"), **STAGE3}


def by_settings(rows: list) -> list[tuple[dict, list]]:
  """Model rows grouped by their settings, in the order first seen: a group
  is read alone, never pooled with another (#481)."""
  groups: dict[str, tuple[dict, list]] = {}
  for r in rows:
    s = settings_of(r)
    groups.setdefault(json.dumps(s, sort_keys=True), (s, []))[1].append(r)
  return list(groups.values())


def _passed_at(r: dict) -> int | None:
  """The round (0 the first) whose fit passed every bar, or None."""
  for i, rd in enumerate(_get(r, "model", "rounds") or []):
    if not rd.get("poor"):
      return i
  return None


def rate(rows: list, max_tokens: int) -> dict:
  """Stage 1's reading of one settings group's model rows (#481), by our
  grading: by round (1 the first), the set-outs that had passed their bars
  and those that had found the lid's hinge, each a share with its Wilson
  interval, and the answers that round with the lid's hinge among them;
  the passes with no lid's hinge (FALSE: a false success, the costly
  error); the answers past `OLD_TOKENS` that came with a document (what the
  larger budget rescued), and of those with none, how many the cap cut off
  (`cut`), how many stopped short of it (`short`: a provider ended them)
  and how many never came (`lost`)."""
  most = max((len(_get(r, "model", "rounds") or []) for r in rows), default=0)
  by_round = []
  for i in range(most):
    got = [r for r in rows if len(_get(r, "model", "rounds") or []) > i]
    by_round.append({
      "round": i + 1,
      "passed": em.share_interval([_passed_at(r) is not None and _passed_at(r) <= i
                                   for r in rows]),
      "found": em.share_interval([any(_lid_right(r, j) for j in range(i + 1)) for r in rows]),
      "answers": len(got), "foundThere": sum(1 for r in got if _lid_right(r, i))})
  passes, false = [], []
  for r in rows:
    i = _passed_at(r)
    if i is not None:
      passes.append(r["k"])
      if not _lid_right(r, i):
        false.append(r["k"])
  turns = [t for r in rows for t in _get(r, "model", "turns") or []]
  return {"n": len(rows), "byRound": by_round, "passes": passes, "false": false,
          "answers": len(turns),
          "rescued": sum(1 for t in turns if (t.get("tokensOut") or 0) > OLD_TOKENS
                         and t.get("document") is not None),
          **_blanks(turns, max_tokens)}


def _blanks(turns: list, max_tokens: int) -> dict:
  """The answers that came with no document: at the cap, short of it, or
  with no tokens at all."""
  out = [t.get("tokensOut") or 0 for t in turns if t.get("document") is None]
  return {"cut": sum(1 for n in out if n >= 0.99 * max_tokens),
          "short": sum(1 for n in out if 0 < n < 0.99 * max_tokens),
          "lost": sum(1 for n in out if n == 0)}


def first_pass(rows: list) -> tuple[dict | None, int]:
  """Of several imaginings of one set-out, in the order they were imagined
  (`_batch`, a summary's own tag), the first that passed its bars -- the
  robot's own measure, never the truth -- and how many it took: one stops
  at its pass, so none after it is ever imagined. None and all of them,
  where none passed (Ben, #481: the first pass of three)."""
  ordered = sorted(rows, key=lambda r: r.get("_batch", 0))
  for i, r in enumerate(ordered):
    if _passed_at(r) is not None:
      return r, i + 1
  return None, len(ordered)


def first_of(rows: list) -> dict | None:
  """#481's first fallback, the first pass of several imaginings: over the
  set-outs one settings imagined more than once (a set-out's under its
  `k`), how often one passed (`first_pass`), the passes kept with no lid's
  hinge, and the imaginings it took a set-out; None where none was
  imagined twice."""
  by_k: dict[int, list] = {}
  for r in rows:
    by_k.setdefault(r["k"], []).append(r)
  many = [rs for rs in by_k.values() if len(rs) > 1]
  if not many:
    return None
  kept = [first_pass(rs) for rs in many]
  return {"setOuts": len(many), "of": sorted({len(rs) for rs in many}),
          "passed": em.share_interval([r is not None for r, _ in kept]),
          "false": [r["k"] for r, _ in kept if r is not None and not _lid_right(r, _passed_at(r))],
          "imaginings": sum(n for _, n in kept) / len(kept)}


def _share(iv) -> str:
  return "--" if iv is None else (f"{iv['k']}/{iv['n']} = {iv['share']:.2f} "
                                  f"[{iv['lo']:.2f}, {iv['hi']:.2f}]")


def _rate_tables(title: str, ms: list, errors: int) -> None:
  """Stage 1's tables for one settings group (`rate`), its bar read, and
  what a set-out cost."""
  s = settings_of(ms[0])
  got = rate(ms, s["maxTokens"])
  outs = len({r["k"] for r in ms})
  print(f"\n== {title}: the rate, {got['n']} imaginings of {outs} set-outs"
        + (f" ({errors} more ended in an error, not counted)" if errors else ""))
  print("  by round: passed by it; the lid's hinge found by it; answers there, the lid's "
        "hinge among them")
  for b in got["byRound"]:
    print(f"    {b['round']}: {_share(b['passed'])}; {_share(b['found'])}; {b['answers']} "
          f"answers, {b['foundThere']}")
  last = got["byRound"][-1]["passed"] if got["byRound"] else None
  print(f"  false passes (no lid's hinge where it passed): {len(got['false'])} of "
        f"{len(got['passes'])}" + (f", set-outs {got['false']}" if got["false"] else ""))
  if last is not None:
    print(f"  the bar, {BAR:.0%} by the last round: {_share(last)}: "
          f"{'MET' if last['share'] >= BAR else 'NOT MET'} (over {last['n']} imaginings the "
          f"interval is wide: a coarse decision)")
  first = first_of(ms)
  if first is not None and last is not None:
    n = max(first["of"])
    print(f"  the first pass of {'/'.join(map(str, first['of']))} imaginings, in the order "
          f"imagined: passed {_share(first['passed'])}, false passes {len(first['false'])}"
          + (f" (set-outs {first['false']})" if first["false"] else "")
          + f"; {first['imaginings']:.2f} imaginings a set-out; {n} independent at the rate "
          f"above would pass {1 - (1 - last['share']) ** n:.2f}")
  print(f"  answers {got['answers']}: past {OLD_TOKENS:,} tokens with a document "
        f"{got['rescued']}; none at the cap of {s['maxTokens']:,} {got['cut']}, short of it "
        f"{got['short']}, no tokens at all {got['lost']}")
  tin = [_get(r, "model", "tokensIn") or 0 for r in ms]
  tout = [_get(r, "model", "tokensOut") or 0 for r in ms]
  print(f"  a set-out: tokens in {_iv(tin, '.0f')}")
  print(f"      out {_iv(tout, '.0f')}")
  usd = [None if not r.get("priceUsdPerMtok") else
         (a * r["priceUsdPerMtok"][0] + b * r["priceUsdPerMtok"][1]) / 1e6
         for r, a, b in zip(ms, tin, tout)]
  if any(u is not None for u in usd):
    print(f"      $ {_iv(usd, '.3f')}; ${sum(u for u in usd if u is not None):.2f} in all")
  print(f"      wall s {_iv([r.get('wallS') for r in ms], '.0f')}")
  # who answered: a policy's providers differ in speed and in how they reason
  who: dict[str, dict] = {}

  def of(name):
    return who.setdefault(name or "no provider (the router's own refusal, or unsaid)",
                          {"turns": [], "rates": [], "graded": 0, "found": 0})
  for r in ms:
    for t in _get(r, "model", "turns") or []:
      w = of(t.get("provider"))
      w["turns"].append(t)
      if t.get("wallS"):
        w["rates"].append((t.get("tokensOut") or 0) / t["wallS"])
    for i, rd in enumerate(_get(r, "model", "rounds") or []):
      if _get(r, "rounds", i) is not None:
        w = of(_get(rd, "turn", "provider"))
        w["graded"] += 1
        w["found"] += _lid_right(r, i)
  for name, w in sorted(who.items()):
    speed = "--" if not w["rates"] else f"{float(np.median(w['rates'])):.0f}"
    b = _blanks(w["turns"], s["maxTokens"])
    print(f"  {name}: {len(w['turns'])} answers, {speed} tokens/s (the median); none at the "
          f"cap {b['cut']}, short of it {b['short']}; the lid's hinge in {w['found']} of its "
          f"{w['graded']} graded answers")


def _author_tables(title: str, ms: list) -> None:
  """The structures one settings group's author wrote, round by round, and
  its first answers' ranges, read by us."""
  print(f"\n== {title}: the author, {len(ms)} set-outs, "
        f"{sum(1 for r in ms if _get(r, 'model', 'kept') is not None)} with a model")
  rounds = [len(_get(r, "model", "rounds") or []) for r in ms]
  print(f"  rounds: {dict(sorted({n: rounds.count(n) for n in set(rounds)}.items()))}; repairs "
        f"{sum(_get(r, 'model', 'repairs') or 0 for r in ms)}")
  toks = [(_get(r, "model", "tokensIn") or 0, _get(r, "model", "tokensOut") or 0) for r in ms]
  print(f"  tokens in {sum(t[0] for t in toks)}, out {sum(t[1] for t in toks)}")
  # an author's ranges where its first answer found the lid's hinge: on
  # any other part, the lid's truth is no truth of that part's
  found = [r for r in ms if _lid_right(r, 0)]
  print(f"  ranges, the {len(found)} first answers that found the lid's hinge:")
  for key in ("mass", "stiffness", "slack", "damping", "friction", "release"):
    first = [(_get(r, "rounds", 0, "ranges", key)) for r in found]
    held = em.share_interval([None if x is None else x["holds"] for x in first])
    widths = [None if x is None else x["width"] for x in first]
    if held:
      print(f"  first answer's {key} range holds the truth {held['k']}/{held['n']} "
            f"[{held['lo']:.2f}, {held['hi']:.2f}]; width {_iv(widths, '.3g')}")
  # the structure each round wrote, read by us: a hinge where the lid's is
  # (its line within 20 mm and 5 deg), the handle on a pin of its own, a
  # catch on the lid's hinge
  for i in range(max(rounds, default=0)):
    got = [r for r in ms if len(_get(r, "model", "rounds") or []) > i]
    lid = [bool(_lid_right(r, i)) for r in got]
    catch = [bool(_lid_right(r, i) and _get(r, "rounds", i, "graded", "model", "release"))
             for r in got]
    pin = [_has_pin(_get(r, "model", "rounds", i, "turn", "document")) for r in got]
    print(f"  round {i}: {len(got)} answers; the lid's hinge {sum(lid)}, a catch on it "
          f"{sum(catch)}, the handle on a pin {sum(pin)}")
  for r in ms:
    kept = _get(r, "model", "kept")
    structure = []
    for rd in _get(r, "model", "rounds") or []:
      doc = _get(rd, "turn", "document") or {}
      structure.append(f"{len(doc.get('parts', []))}p/"
                       f"{'+'.join(j.get('type', '?')[0] for j in doc.get('joints', []))}/"
                       f"{len(doc.get('catches', []))}c"
                       f"{'*' if rd.get('poor') else ''}")
    print(f"  k {r['k']:3d} catch {r['catchN']:.2f} kept {kept}: {' -> '.join(structure)}")


def _title(s: dict) -> str:
  levers = ", ".join(s.get("levers") or ()) or "no levers"
  return (f"model by {s.get('author')}, {s['maxRounds']} rounds, {s['maxTokens']:,} tokens, "
          f"{levers}")


def summary(into: Path, probes: Path | None, also: tuple = ()) -> None:
  """The tables: each condition apart, each model's settings apart, clean
  and flagged set-outs apart; `also` more batches' directories, whose
  rows of one settings are more imaginings of the same set-outs."""
  rows = sorted(({**json.loads(p.read_text()), "_batch": i}
                 for i, d in enumerate((into, *also)) for p in d.glob("*_*.json")),
                key=lambda r: (r.get("k", -1), r.get("condition", ""), r["_batch"]))
  flagged = set()
  replays = {}
  if probes is not None and (probes / "rows.jsonl").exists():
    for line in open(probes / "rows.jsonl"):
      r = json.loads(line)
      if r.get("twisted"):
        flagged.add(r["k"])
      replays[r["k"]] = r.get("replay")
  models = [r for r in rows if r.get("condition") == "model"]
  for s, group in by_settings(models):
    ms = [r for r in group if "error" not in r]
    title = _title(s)
    every = _graded(ms, "model")
    # a model that passed its bars, and one that found the lid's hinge but
    # did not: apart, never pooled
    def kept_round(r):
      return _get(r, "model", "rounds", _get(r, "model", "kept"))
    ok = [x for x in every if not kept_round(x[0])["poor"]]
    lid = [x for x in every if kept_round(x[0])["poor"]
           and _lid_right(x[0], _get(x[0], "model", "kept"))]
    for name, sel_ in ((", passed", ok), (", the lid's hinge but poor", lid)):
      for label, sel in (("clean", [x for x in sel_ if x[0]["k"] not in flagged]),
                         ("flagged", [x for x in sel_ if x[0]["k"] in flagged])):
        if sel:
          _tables(f"{title}{name}, {label} set-outs", sel, replays)
    if ms:
      _author_tables(title, ms)
      _rate_tables(title, ms, len(group) - len(ms))
  every = _graded(rows, "reference")
  for label, sel in (("clean", [x for x in every if x[0]["k"] not in flagged]),
                     ("flagged", [x for x in every if x[0]["k"] in flagged])):
    if sel:
      _tables(f"reference, {label} set-outs", sel, replays)
  _leak_tables(rows, flagged)


def demo(args) -> None:
  """Set-out `args.demo` flown and imagined, recorded (the module docstring)."""
  import sys
  import mujoco
  from pluggybot.activity import chest as ch
  from pluggybot.activity.base import ActivitySet
  from pluggybot.home import world as home
  from pluggybot.imagination import ghost
  from pluggybot.imagination.author import Author
  from pluggybot.legs import body as qb
  from pluggybot.lifecycle import world_config
  from pluggybot.mind.llm import HFClient, stream_fetch
  from pluggybot.telemetry import scene as ts
  from pluggybot.telemetry.protocol import ROBOT_ROOT
  from pluggybot.telemetry.recorder import TelemetryRecorder
  sys.path.insert(0, str(Path(__file__).parent))
  import probe_chest as pc
  k = args.demo
  so = ep.set_out(k)
  m = ep.spec_of(so).compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  body = qb.QuadBody(m, d, realtime=False, grid_bounds=home.GRID_BOUNDS)
  mis = body.mission
  acts = ActivitySet([ch.Chest(m, d, so.lid)])
  said = {"state": "PROBE", "status": "probing the chest"}

  def status() -> dict:
    return {**said, "posture": mis.posture,
            "battery": {"frac": 1.0, "watts": 0.0, "charging": False}}
  record = Path(args.record)
  rec = TelemetryRecorder(m, d, str(record), model_name=DEMO_WORLD, status_fn=status,
                          activities=acts, grid=mis.grid)
  mis.step_hooks += [acts.step_hook(m, d), rec.step_hook]
  pc.mount(body, *so.start)
  out = body.run(pr.Probe(mis).routine())
  print(json.dumps({"k": k, "ok": out.ok, "why": out.why, "simS": round(float(d.time), 1)}),
        flush=True)
  if not out.ok:
    rec.close()
    raise SystemExit(f"the probe did not get through: {out.why}")
  said.update(state="IMAGINE", status="imagining the chest")
  body.run(mis.rest_routine())
  s = pr.sizes(out.record, out.guess)
  if isinstance(s, str):
    rec.close()
    raise SystemExit(f"the box was not measured: {s}")
  from pluggybot.legs.imagined import felt
  felt_rows = felt(out.record, out.friction)
  remotes = args.remote.split(",") if args.remote else []

  def attempt(i: int):
    """One imagining from nothing: a fresh author, its own workers."""
    argv_w = None
    if remotes:
      r = remotes[i % len(remotes)]
      open_master(r, 100 + i)
      argv_w = remote_argv(r, 100 + i)
    author = Author(HFClient(timeout=READ_S, fetch=stream_fetch), args.author)
    with Imaginations(args.workers, argv=argv_w) as pool:
      return mm.imagine(author, out.record, pr.phases_of(out), pr.fit_rows(out),
                        s.as_dict(), s.origin, s.yaw, pr.did(out, s),
                        pr.exposed(out.picture) if out.picture else None, pool,
                        felt=felt_rows)
  t0 = time.time()
  with ThreadPoolExecutor(args.attempts) as threads:
    models = list(threads.map(attempt, range(args.attempts)))
  # the first in ORDER whose kept round passed, never the fastest: a choice
  # the threads' timing cannot move; none passed, the first that kept one
  passed = [i for i, m in enumerate(models) if m.round is not None and not m.round.poor]
  made = [i for i, m in enumerate(models) if m.round is not None]
  chosen = (passed or made or [0])[0]
  model = models[chosen]
  print(json.dumps({"imagined": model.kept is not None, "attempts": args.attempts,
                    "chosen": chosen, "passed": passed, "rounds": len(model.rounds),
                    "wallS": round(time.time() - t0, 1)}), flush=True)
  body.run(body.hold_routine(THINK_S))
  kept = model.round
  if kept is not None:
    rec.emit(ghost.message(model.document, kept.fitted.readings, robot=ROBOT_ROOT,
                           t=float(d.time), t0=out.t0, dt=out.record.dt,
                           what=str((kept.turn.answer or {}).get("what") or ""),
                           extra={"rounds": len(model.rounds), "revisions": model.revisions,
                                  "attempts": {"made": args.attempts, "this": chosen,
                                               "passed": len(passed)},
                                  "residualN": {p["phase"]: round(p["rmsN"], 4)
                                                for p in kept.judged}}))
    said.update(state="IMAGINED", status="imagined the chest")
  else:
    said.update(state="IMAGINED", status="could not imagine the chest")
  body.run(body.hold_routine(AFTER_S))
  rec.close()
  # the demo's world, as a replayer draws it: its scene and its textures
  meta_path = Path(world_config("home_quad")["meta"])
  meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
  scene = ts.scene_dict(m, DEMO_WORLD, meta=meta)
  (record.parent / f"scene.{DEMO_WORLD}.json").write_text(json.dumps(scene, indent=1) + "\n")
  ts.export_textures(m, record.parent / "textures")
  (record.parent / f"model_{k}.json").write_text(json.dumps(_clean(
    {"chosen": chosen, "attempts": [m.as_dict() for m in models]})))
  print(f"{record}: the probe, the thinking and the ghost; the scene and the model beside it")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--probes")
  ap.add_argument("--into")
  ap.add_argument("--n", type=int, default=None)
  ap.add_argument("--ks", default=None, help="set-outs by number, a comma list")
  ap.add_argument("--from", dest="first", type=int, default=0)
  ap.add_argument("--conditions", default="model")
  ap.add_argument("--workers", type=int, default=5)
  ap.add_argument("--slots", type=int, default=2)
  ap.add_argument("--remote", default=None)
  ap.add_argument("--author", default=os.environ.get("PLUGGY_MODEL") or DEPLOYED)
  ap.add_argument("--summary", action="store_true")
  ap.add_argument("--also", default="", help="--summary: more batches' directories, a comma list")
  ap.add_argument("--demo", type=int, default=None)
  ap.add_argument("--attempts", type=int, default=1)
  ap.add_argument("--record", default=None)
  args = ap.parse_args(argv)
  if args.demo is not None:
    if not args.record:
      ap.error("--demo records: --record PATH")
    demo(args)
    return
  into = Path(args.into)
  if args.summary:
    if not args.probes:
      ap.error("--summary reads which set-outs are flagged off --probes' rows: give it")
    summary(into, Path(args.probes),
            tuple(Path(d).expanduser() for d in args.also.split(",") if d))
    return
  args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
  if any(c not in ("model", "reference", "leak") for c in args.conditions):
    ap.error("a condition is model, reference or leak")
  into.mkdir(parents=True, exist_ok=True)
  if "model" in args.conditions:
    # what the author's tokens cost, off the router's catalogue once (None:
    # unknown, never zero); each row carries it beside its settings
    from pluggybot.mind.llm import HFClient
    args.price = HFClient(timeout=30.0).pricing(args.author)
    print(json.dumps({**settings(args.author), "priceUsdPerMtok": args.price}), flush=True)
  kept = sorted(int(p.stem.split("_")[1]) for p in Path(args.probes).glob("probe_*.npz"))
  ks = [k for k in kept if k >= args.first][:args.n]
  if args.ks:
    ks = [k for k in kept if k in {int(x) for x in args.ks.split(",")}]
  # a slot is a set-out at once, with its workers here or on one pod
  if args.workers > 10 and args.remote:
    ap.error("a slot's workers share one ssh connection, ten channels at most")
  slots: queue.SimpleQueue = queue.SimpleQueue()
  where = args.remote.split(",") if args.remote else [None]
  for r in where:
    for i in range(args.slots):
      if r is None:
        slots.put(None)
        continue
      open_master(r, i)
      slots.put(remote_argv(r, i))

  def slotted(k):
    argv_w = slots.get()
    try:
      return one(k, args, argv_w)
    finally:
      slots.put(argv_w)
  lock = threading.Lock()
  with ThreadPoolExecutor(len(where) * args.slots) as threads, \
       open(into / "rows.jsonl", "a") as f:
    for rows in threads.map(slotted, ks):
      with lock:
        for r in rows if isinstance(rows, list) else [rows]:
          f.write(json.dumps(r) + "\n")
          f.flush()
          print(json.dumps({k_: r.get(k_) for k_ in ("k", "condition", "error", "wallS")}),
                flush=True)


if __name__ == "__main__":
  main()
