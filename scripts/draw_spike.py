"""Drawing on a whiteboard with the quadruped's arm (issue #406; SimNotes,
"Drawing on legs"): what a figure comes out as, and why.

  --sway [--stance S,...]   a pen held pressed on the board 30 s in each
                            stance: how far the ink point wanders (the
                            stance table)
  (default)                 the served body, the pen on its fork, the board's
                            tags found: walk to the board, lie down, find
                            the face by touch, draw, stand up and back out;
                            the figure's form error and inked fraction off
                            the WORLD (the plotter's trace), and the
                            calibration's plane against the truth
  --figure NAME             square (default), circle, house, tree, sun,
                            robot, or answer:NN
  --board NAME              whiteboard_a (default) or whiteboard_b
  --stance stand            draw on its feet, the walking policy holding it
  --n N [--jobs J]          N flights, each from a start jittered round the
                            look point (0.2 m, 15 deg)

Filmstrip `draw_spike.png`: the board's ink, the figure asked for over it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool

import mujoco
import numpy as np


def _world(stance_board: str = "whiteboard_a"):
  from pluggybot.home import world as home
  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=home.GRID_BOUNDS)
  return model, data, body


def _board(name: str):
  from pluggybot.home import world as home
  from pluggybot.tools.drawing import Board
  return Board.from_meta(home.BOARDS[name])


def mount_pen(body) -> None:
  """The pen seated on the fork at the carry pose, as a fetch leaves it,
  and carried: a placement, not a swap (the swap is `arm_spike.py`'s)."""
  from pluggybot.legs import arm as am
  from pluggybot.rack.coupling import PEG_ABOVE_BODY
  mis, m, d = body.mission, body.model, body.data
  mis.arm.aim(*am.CARRY_Q)
  body.run(body.hold_routine(2.0))
  seat = m.site(mis.handle.el("arm_seat")).id
  adr = m.jnt_qposadr[m.joint("module_pen_free").id]
  dof = m.jnt_dofadr[m.joint("module_pen_free").id]
  rot = d.xmat[mis.root].reshape(3, 3)
  _, _, yaw = mis.true_pose()
  peg = d.site_xpos[seat] + rot @ np.array([0, 0, mis.arm_spec.fork.seat_rise() + 0.0003])
  d.qpos[adr:adr + 3] = peg - [0, 0, PEG_ABOVE_BODY]
  d.qpos[adr + 3:adr + 7] = [math.cos((yaw + math.pi) / 2), 0, 0, math.sin((yaw + math.pi) / 2)]
  d.qvel[dof:dof + 6] = 0.0
  mujoco.mj_forward(m, d)
  mis.carry("module_pen")
  body.run(body.hold_routine(1.0))


def program_for(figure: str):
  from pluggybot.tools import strokes
  from pluggybot.tools.drawing import Envelope
  if figure.startswith("answer:"):
    return strokes.program("answer", text=figure.split(":", 1)[1])
  prog = strokes.program(figure)
  env = Envelope.for_board(_board("whiteboard_a"))
  return prog if prog.fits(env) else prog.fitted(env)


def fly(args) -> dict:
  """One drawing (the module docstring)."""
  k, figure, board_name, stance = args
  from pluggybot.legs.draw import LOOK_M
  model, data, body = _world()
  mis = body.mission
  board = _board(board_name)
  rng = np.random.default_rng(406 + k)
  nx, ny = board.normal
  fx, fy = board.x + nx * board.half[0], board.y + ny * board.half[0]
  jx, jy = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)) if k else (0.0, 0.0)
  jh = math.radians(rng.uniform(-15, 15)) if k else 0.0
  body.start_at(fx + nx * (LOOK_M + 0.3) + jx, fy + ny * (LOOK_M + 0.3) + jy,
                board.heading + jh)
  mount_pen(body)
  mis._look_at_board(board_name)
  inks = []
  rec = body.run(mis.draw_routine(board_name, board, program_for(figure), patience=600.0,
                                  on_stroke=lambda i, pts, name: inks.append(pts),
                                  stance=stance))
  out = {"k": k, "figure": figure, "board": board_name, "stance": stance,
         "why": rec["why"], "s": rec.get("seconds"), "walkIn": rec.get("walkIn"),
         "falls": mis.falls}
  for key in ("strokes", "strokes_drawn", "inked_fraction", "travel_ink_fraction",
              "form_rms_mm", "form_max_mm", "shape_rms_mm", "offset_mm"):
    if rec.get(key) is not None:
      out[key] = round(rec[key], 3) if isinstance(rec[key], float) else rec[key]
  cal = rec.get("cal") or {}
  out["cal"] = {k2: cal.get(k2) for k2 in ("probes", "rmsMm", "yawDeg", "ok")}
  # the calibration against the TRUTH: where the plane says the face is,
  # against where the tip touched it (the instrument's, never the robot's)
  p = mis.last_plotter
  if p is not None and p._home is not None:
    out["homeMm"] = [round(v * 1000, 1) for v in p._home]
  out["pose"] = [round(v, 3) for v in mis.true_pose()]
  out["est"] = rec.get("est")
  out["stillCarrying"] = mis.carrying
  out["inks"] = inks
  body.close()
  return out


def sway_one(args) -> dict:
  """The stance table's one row: the pen pressed on the board's middle and
  held 30 s."""
  stance, seconds = args
  from pluggybot.legs.draw import LIE_M, STAND_M
  from pluggybot.tools.drawing import BOARD_PROUD_M, BoardEstimate, PenPlotter
  model, data, body = _world()
  mis = body.mission
  board = _board("whiteboard_a")
  nx, ny = board.normal
  fx, fy = board.x + nx * board.half[0], board.y + ny * board.half[0]
  # ...where the walk in stops: `STAND_M` out of the wall its tags are on
  out = STAND_M - BOARD_PROUD_M
  body.start_at(fx + nx * out, fy + ny * out, board.heading)
  mount_pen(body)
  mis.working = True
  if stance == "lie":
    body.run(mis.rest_routine())
  p = PenPlotter(mis, board)
  est = BoardEstimate(x_face=(LIE_M if stance == "lie" else STAND_M) - BOARD_PROUD_M,
                      y_mid=0.0, z_mid=board.z - mis._height())
  cal = body.run(p.calibrate_routine(est))
  x = p._x_at(0.0, cal["zHome"], 0.010)
  body.run(p.move_routine(x, cal["zHome"], speed=0.02))
  body.run(p._still_routine(1.0))
  rows = []
  t0 = data.time
  p0 = mis.true_pose()
  while data.time - t0 < seconds:
    body.run(p._still_routine(0.1))
    lat, h = p.pen_board()
    x_, y_, yaw = mis.true_pose()
    rows.append((lat, h, x_ - p0[0], y_ - p0[1], yaw - p0[2]))
  a = np.array(rows)
  out = {"stance": stance, "posture": mis.posture,
         "tipLatMm": round(float(np.ptp(a[:, 0]) * 1000), 2),
         "tipHMm": round(float(np.ptp(a[:, 1]) * 1000), 2),
         "torsoMm": round(float(math.hypot(a[-1, 2], a[-1, 3]) * 1000), 2),
         "yawDeg": round(math.degrees(float(a[-1, 4])), 3), "falls": mis.falls}
  body.close()
  return out


def filmstrip(results, out: str = "draw_spike.png", px_per_m: float = 2500.0) -> None:
  """Each flight's ink, as the board book would paint it: +lat is the
  viewer's LEFT."""
  from PIL import Image, ImageDraw
  tile = int(0.16 * px_per_m)
  img = Image.new("L", (tile * len(results), tile + 14), 255)
  draw = ImageDraw.Draw(img)
  for j, r in enumerate(results):
    ox, oy = j * tile + tile // 2, tile // 2 + 14
    for pts in r["inks"]:
      if len(pts) > 1:
        draw.line([(ox - lat * px_per_m, oy - h * px_per_m) for lat, h in pts], fill=0, width=2)
    draw.text((j * tile + 4, 1), f"{r['figure']} #{r['k']} {r.get('form_rms_mm', '-')} mm", fill=0)
  img.save(out)


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--sway", action="store_true")
  ap.add_argument("--stance", default=None)
  ap.add_argument("--figure", default="square")
  ap.add_argument("--board", default="whiteboard_a")
  ap.add_argument("--n", type=int, default=1)
  ap.add_argument("--jobs", type=int, default=3)
  ap.add_argument("--seconds", type=float, default=30.0)
  a = ap.parse_args(argv)
  if a.sway:
    stances = (a.stance or "lie,stand").split(",")
    with Pool(min(a.jobs, len(stances))) as pool:
      for r in pool.imap_unordered(sway_one, [(s, a.seconds) for s in stances]):
        print(json.dumps(r), flush=True)
    return
  stance = a.stance or "lie"
  with Pool(min(a.jobs, a.n)) as pool:
    results = []
    for r in pool.imap_unordered(fly, [(k, a.figure, a.board, stance) for k in range(a.n)]):
      results.append(r)
      print(json.dumps({k: v for k, v in r.items() if k != "inks"}), flush=True)
  results.sort(key=lambda r: r["k"])
  filmstrip(results)
  forms = [r["form_rms_mm"] for r in results if r.get("form_rms_mm") is not None]
  if forms:
    print(f"{a.figure} on {a.board}, {stance}: drew {len(forms)}/{len(results)}, form "
          f"median {np.median(forms):.2f} mm (worst {max(forms):.2f}), inked "
          f"{np.median([r['inked_fraction'] for r in results if 'inked_fraction' in r]):.0%}")


if __name__ == "__main__":
  sys.exit(main())
