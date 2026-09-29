#!/usr/bin/env python
"""Places, not coordinates (issue #419): where a plate's tag reads from, and
whether a robot that has never seen the lab finds a plate and presses it.

    MUJOCO_GL=egl uv run python scripts/places_spike.py --tags
        the nose camera's envelope on the lab's signs: a quadruped standing
        at each distance and angle off a sign's face, the decode, where it
        put the tag and what its own rotation read -- and, with --flat, the
        same tag lying flat on the floor, the candidate the signs beat
    MUJOCO_GL=egl uv run python scripts/places_spike.py --find --n 8 --parallel 4
        N fresh quadrupeds, each from its start in its own process, sent to
        find the feed plate by its tag and press it, searching round the
        facility's address -- the data's own (`home/places.json`), or with
        --n > 1 an error of --error metres in N directions round the house's
        middle -- and how each went: found or why not, the time, the
        guesses, viewpoints and looks round, the press, every plate's
        presses (the shock plate must read 0), the belief against the truth
    ... --again        then back to the start, and find and press it again
                       from the place it remembers
    ... --tag 35       another plate (a shock press is then the point)
    ... --from dock    start lying on the dock instead of at `start`
    ... --out f.json   the records

Every flight is deterministic: the same start, address and plate fly the
same walk, so a success RATE is over addresses and starts, never repeats.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The envelope's grid: metres from the camera, degrees off the sign's face.
DISTANCES = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
ANGLES = (0, 10, 20, 30, 40, 55, 70)
#: The nose camera ahead of the torso's centre, m (`legs.model`'s mount).
CAMERA_AHEAD_M = 0.222


def envelope(flat: bool = False) -> list[dict]:
  """Stand a quadruped in front of the feed plate's sign (or a tag flat on
  open floor) at every distance and angle, facing it, and decode once."""
  import mujoco

  from pluggybot.activity import cage
  from pluggybot.home import world as home
  from pluggybot.legs import dock as dk
  from pluggybot.legs import world as lw
  from pluggybot.rack.tags import PLATE_TAG_SIZE, TagDetector, plate_half_extent
  tag = cage.PLATE_TAGS["feed"]
  spec = lw.home_spec(first_at=(12.5, -4.0))
  if flat:
    # the same tag on the middle street's open floor, 2 mm over its slab
    half = plate_half_extent(0.32)
    child = mujoco.MjSpec.from_string(
      f"<mujoco><worldbody><body name='flat_tag' pos='12.5 -2 0'>"
      f"<geom type='box' size='{half} {half} 0.001' pos='0 0 0.004' contype='0' "
      f"conaffinity='0' material='tagmat{tag}'/></body></worldbody></mujoco>")
    spec.attach(child, prefix="", frame=spec.worldbody.add_frame())
  model = spec.compile()
  data = mujoco.MjData(model)
  from pluggybot.rack import tags as tg
  if flat:
    tg.TAG_SIZES[tag] = 0.32
  det = TagDetector(model, "nav_eye")
  root = model.body("pluggybot").id
  if flat:
    tx, ty, normal = 12.5, -2.0, -math.pi / 2
  else:
    tx, ty = cage.sign_xy(home.LAB_CAGE_XY, "feed")
    ty -= 0.001
    normal = -math.pi / 2
  out = []
  for dist in DISTANCES:
    for off in ANGLES:
      a = normal - math.radians(off)
      cx, cy = tx + dist * math.cos(a), ty + dist * math.sin(a)
      yaw = math.atan2(ty - cy, tx - cx)
      x, y = cx - CAMERA_AHEAD_M * math.cos(yaw), cy - CAMERA_AHEAD_M * math.sin(yaw)
      lw.stand(model, data, "", x, y, yaw)
      dets = det.detect(data)
      row = {"dist": dist, "off": off, "seen": tag in dets}
      if tag in dets:
        d = dets[tag]
        hx, hy = dk.seen_from(model, data, {tag: d}, "nav_eye", root)[tag]
        c, s = math.cos(yaw), math.sin(yaw)
        row["errMm"] = round(1000 * math.hypot(x + c * hx - s * hy - tx,
                                               y + s * hx + c * hy - ty), 1)
        row["yawDeg"] = round(math.degrees(d["yaw"]), 1)
        row["yawErrDeg"] = round(abs(abs(math.degrees(d["yaw"])) - off), 1)
      out.append(row)
  det.close()
  if flat:
    tg.TAG_SIZES[tag] = PLATE_TAG_SIZE
  return out


def print_envelope(rows: list[dict], flat: bool) -> None:
  print(("a 320 mm tag flat on the floor" if flat else "the feed plate's sign, a 120 mm tag")
        + ": decoded (+) or not (-), by metres from the camera and degrees off its face")
  print("  deg  " + " ".join(f"{d:>4}" for d in DISTANCES))
  for off in ANGLES:
    cells = [r for r in rows if r["off"] == off]
    print(f"  {off:>3}  " + " ".join(f"{'+' if r['seen'] else '-':>4}" for r in cells))
  seen = [r for r in rows if r["seen"]]
  if seen:
    errs = sorted(r["errMm"] for r in seen)
    print(f"  position: median {errs[len(errs) // 2]} mm, worst {errs[-1]} mm "
          f"over {len(seen)} decodes")
    if not flat:
      for off in ANGLES:
        ys = [r["yawErrDeg"] for r in seen if r["off"] == off]
        if ys:
          print(f"  own rotation {off:>2} deg off: worst {max(ys)} deg wrong "
                f"({sum(1 for y in ys if y > 5)} of {len(ys)} past 5)")


def fly_one(tag: int, near: tuple[float, float], start: str, again: bool,
            patience: float) -> dict:
  """One fresh robot: a look round where it starts, a find, a press; with
  `again`, back to the start and both once more."""
  import mujoco

  from pluggybot.activity.cage import PLATE_TAGS, Cage
  from pluggybot.home import world as home
  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from pluggybot.procedure.steps import PRESS_PATIENCE_S
  cfg = world_config(QUAD_HOME)
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=cfg["grid_bounds"])
  m = body.mission
  lab = next(z for z in home.ZONES if z["name"] == "lab")
  cage = Cage(model, data, room=(tuple(lab["min"]), tuple(lab["max"])))
  m.step_hooks.append(lambda: cage.sense(model, data))
  sx, sy, syaw = m.charge_standoff() if start == "dock" else cfg[start]
  m.start_at(sx, sy, syaw)
  w0 = time.perf_counter()
  body.run(body.look_around_routine())
  names = {v: k for k, v in PLATE_TAGS.items()}

  def one() -> dict:
    t0 = float(data.time)
    before = dict(cage.counts)
    found = body.run(body.find_tag_routine(tag, near=near, patience=patience))
    pressed = (body.run(body.press_plate_routine(tag, patience=PRESS_PATIENCE_S))
               if found.get("found")
               else {"pressed": False, "why": "not found"})
    tx, ty, _ = m.true_pose()
    bx, by, _ = m.pose
    return {"found": bool(found.get("found")), "why": found.get("why"),
            "remembered": bool(found.get("remembered")),
            "findS": found.get("seconds"), "guesses": found.get("guesses"),
            "viewpoints": found.get("viewpoints"), "arounds": found.get("arounds"),
            "pressed": bool(pressed.get("pressed")), "pressWhy": pressed.get("why"),
            "stop": (pressed.get("attempts") or [{}])[-1].get("stop"),
            "seconds": round(float(data.time) - t0, 1),
            "presses": {names[PLATE_TAGS[n]]: cage.counts[n] - before[n]
                        for n in PLATE_TAGS},
            "beliefOffM": round(math.hypot(bx - tx, by - ty), 3)}

  first = one()
  second = None
  if again:
    body.run(body.go_to_routine(sx, sy, timeout=patience))
    second = one()
  places = {p.tag: [round(p.x, 3), round(p.y, 3), p.n] for p in m.places}
  out = {"tag": tag, "near": list(near), "from": start, "first": first,
         "again": second, "places": places, "falls": m.falls,
         "wallS": round(time.perf_counter() - w0, 1), "simS": round(float(data.time), 1)}
  body.close()
  return out


def addresses(n: int, error: float) -> list[tuple[float, float]]:
  """The data's own address for one flight; else N directions round the
  facility's middle, `error` metres out."""
  from pluggybot.home import places
  if n <= 1:
    a = places.address("facility")
    return [(a["x"], a["y"])]
  mx, my = places.house_middle("facility")
  return [(round(mx + error * math.cos(2 * math.pi * k / n), 2),
           round(my + error * math.sin(2 * math.pi * k / n), 2)) for k in range(n)]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--tags", action="store_true", help="the decode envelope")
  ap.add_argument("--flat", action="store_true", help="...of a tag flat on the floor")
  ap.add_argument("--find", action="store_true", help="fly the find and the press")
  ap.add_argument("--tag", type=int, default=36, help="the plate's tag (36, the feed)")
  ap.add_argument("--n", type=int, default=1, help="flights: address errors round the house")
  ap.add_argument("--error", type=float, default=3.0, help="...this far out, m")
  ap.add_argument("--from", dest="start", default="start",
                  choices=("start", "start2", "dock"))
  ap.add_argument("--again", action="store_true", help="and again, remembered")
  ap.add_argument("--patience", type=float, default=600.0)
  ap.add_argument("--parallel", type=int, default=1)
  ap.add_argument("--out", default=None)
  ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args()
  if args.one:
    spec = json.loads(args.one)
    print(json.dumps(fly_one(spec["tag"], tuple(spec["near"]), spec["from"],
                             spec["again"], spec["patience"])))
    return
  if args.tags or args.flat:
    rows = envelope(flat=args.flat)
    print_envelope(rows, args.flat)
    if args.out:
      Path(args.out).write_text(json.dumps(rows, indent=1))
    return
  if not args.find:
    ap.error("say --tags, --flat or --find")

  def run(near):
    spec = json.dumps({"tag": args.tag, "near": near, "from": args.start,
                       "again": args.again, "patience": args.patience})
    res = subprocess.run([sys.executable, __file__, "--one", spec],
                         capture_output=True, text=True)
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
    if not lines:
      return {"near": list(near), "error": res.stderr[-2000:]}
    return json.loads(lines[-1])

  with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
    recs = list(pool.map(run, addresses(args.n, args.error)))
  for r in recs:
    if "error" in r:
      print(f"{r['near']}: FAILED\n{r['error']}")
      continue
    for label, f in (("first", r["first"]), ("again", r["again"])):
      if f is None:
        continue
      print(f"{str(r['near']):>16} {label:>5}: found {f['found']!s:5} ({f['why']}) in "
            f"{f['findS']:>6} s, {f['guesses']} guesses {f['viewpoints']} viewpoints "
            f"{f['arounds']} looks round; pressed {f['pressed']!s:5} stop {f['stop']}; "
            f"presses {f['presses']}; belief off {f['beliefOffM']} m")
    print(f"{'':>16}        falls {r['falls']}, {r['simS']} sim s in {r['wallS']} s")
  firsts = [r["first"] for r in recs if "first" in r]
  if firsts:
    ok = [f for f in firsts if f["found"] and f["pressed"]]
    times = sorted(f["findS"] for f in firsts if f["found"])
    shock = sum(f["presses"]["shock"] for f in firsts) if args.tag != 35 else None
    print(f"found and pressed {len(ok)} of {len(firsts)}; find "
          + (f"{times[0]}-{times[-1]} s, median {times[len(times) // 2]} s" if times else "-")
          + ("" if shock is None else f"; shock-plate presses {shock}"))
  if args.out:
    Path(args.out).write_text(json.dumps(recs, indent=1))


if __name__ == "__main__":
  main()
