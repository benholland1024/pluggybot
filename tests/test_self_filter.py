"""#762: the self-filter round a held grip (`mission.HELD_DROP_M`, the LIDAR's;
`HELD_DEPTH_M`, the depth camera's; SimNotes, "The self-filter sized by what
a job carries"). Each rule pinned without a flight -- the bare body posed at
the carry pose, its sensors cast at it, a direct call (docs/Testing.md):

  1. the LIDAR's radius lies between two measures: over what a carried
     candle's own returns need at one sigma of its range noise -- its
     farthest point the rings reach at the carry pose and the 3 cm the
     robot's model of itself may place its grip off the true one
     (`GRIP_OFF_M`) -- and under the nearest edge a carried piece's work puts
     in reach: the pass's, beside the carried fork as its walk begins;
  2. so a carried candle rarely stops the walk at it, where at 0.12 it did on
     a fifth of the scans carried; and the pass's edge stops it, where at
     0.25 the edge was hidden and the walk turned into it;
  3. no piece a job carries (`CARRIED`: its drawing, and where and how its
     reference grips it) stops the walk at it, and those grips are the
     references' own: a reference that moves its grip fails here first;
  4. the depth camera's radius covers a cup or a candle whole -- its noise a
     few mm -- and no more: a plate's rim and a fork's head stay in its
     frame, as they did;
  5. each sensor drops at its own radius: the band at the LIDAR's, the depth
     camera's frame at its own.

#736's merge adds its candle and its frame to `CARRIED`, and its candle to
`WHOLE`.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.activity import cube as cb
from pluggybot.activity import dishware as dw
from pluggybot.body import body_for
from pluggybot.challenge import solutions as sol
from pluggybot.evaluation import counter as ct
from pluggybot.evaluation import hands_check as hc
from pluggybot.home import world as home
from pluggybot.humanoid import mission as hmis
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import HandGoal
from pluggybot.perception import depth as pd
from pluggybot.perception import lidar as pl

#: How far the robot's model of itself may place its grip off the true one,
#: m: the placement pin's tolerance
#: (`test_carry.py::test_the_mission_places_a_held_grip_off_its_model_of_itself_and_drops_it_from_its_layer`;
#: in a carried walk it was 2-13 mm, #762's flights).
GRIP_OFF_M = 0.03
#: The radius #762's first round flew, m: wide enough for the forks' heads at
#: two sigmas, it hid the pass's edge (SimNotes).
WIDE_M = 0.25
#: Today's before #762, m (#682's, both sensors').
OLD_M = 0.12

#: A tag's face over the face it is printed on, m (`dishware._film`).
_TAG = dw.FILM
#: The reaches' turns: fingers 45 deg down, jaws across (the cups'); fingers
#: 60 deg down, jaws along -y (the plate's and the fork's).
_CUP_TURN = ((0.707, 0.0, -0.707), (0.0, 1.0, 0.0))
_DOWN60 = ((0.5, 0.0, -0.866), (0.0, -1.0, 0.0))
#: A candle of #736's size -- 70 mm across, 100 mm tall -- gripped as #736's
#: reference grips its own: 2 cm short of its axis and 1.5 cm over its
#: middle, its fingers level.
CANDLE = ("a candle-sized piece", "candle", "left", (-0.02, 0.0, 0.065), (1.0, 0.0, 0.0),
          (0.0, 1.0, 0.0))


def _piece(name):
  """A drawing as a body a test can pose (mocap, no joint)."""
  def add(spec, body_name):
    if name == "cube":
      b = spec.worldbody.add_body(name=body_name, mocap=True)
      b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[cb.CUBE] * 3, pos=[0.0, 0.0, cb.CUBE])
      return
    dw.ATTACH[name](spec, (0.0, 0.0), 0.0, 0.0, 0.3, 0.7, tags=False, name=body_name)
    b = spec.body(body_name)
    for j in list(b.joints):
      spec.delete(j)
    b.mocap = True
  return add


#: EVERY PIECE A JOB MAY CARRY, as its reference grips it: (what, its drawing,
#: the hand, the grip's place in the piece's frame as it is set out -- level,
#: unturned in the stance, z up from what it stands on -- the reach's fingers
#: and jaws there, and each reference's words that put it there, read in its
#: source by `test_the_grips_are_the_references_own`: the kitchen shelf's are
#: the pass's sources rewritten, its grips as the pass's). The pieces are set
#: out as their references find them: the fork with its head ahead of the hand
#: (`counter.PASS_SOURCES`: "its head in").
CARRIED = (
  ("the pass's and the kitchen shelf's cup", "cup", "right", (0.0, 0.0, dw.CUP_H + _TAG - 0.015),
   *_CUP_TURN, ((sol.PASS, 'cz = read("spot.z") - 0.015'), (sol.SHELF, 'cz = read("spot.z") - 0.015'))),
  ("the pass's and the kitchen shelf's plate", "plate", "left",
   (0.0, sol.PASS_PLATE_ACROSS, dw.PLATE_FLOOR_T + _TAG + 0.0384), *_DOWN60,
   ((sol.PASS, 'gz = read("spot.z") + 0.0384'), (sol.SHELF, 'gz = read("spot.z") + 0.0384'))),
  ("the pass's and the kitchen shelf's fork", "fork", "left",
   (0.0, 0.0, 2 * dw.HANDLE[2] + _TAG + 0.0264), *_DOWN60,
   ((sol.PASS, 'gz = read("spot.z") + 0.0264'), (sol.SHELF, 'gz = read("spot.z") + 0.0264'))),
  ("the table check's cup", "cup", "right", (0.0, 0.0, dw.CUP_H + _TAG - 0.03), *_CUP_TURN,
   ((hc.TABLE, 'cz = read("spot.z") - 0.03'),)),
  ("the table check's plate", "plate", "left", (0.0, 0.066, 0.7734 - ct.TABLE_H), *_DOWN60,
   ((hc.TABLE, "gy = py + 0.066 * (1 - i)\n    gz = 0.7734 - 0.005 * i"),)),
  ("the table check's fork", "fork", "left", (0.0, 0.0, 0.7734 - 0.005 - ct.TABLE_H), *_DOWN60,
   ((hc.TABLE, "gz = 0.7734 - 0.005 * i"),)),
  ("the cube check's cube", "cube", "left", (0.01, 0.0, cb.CUBE + 0.003), *_DOWN60[:1],
   (0.0, 1.0, 0.0), ((hc.PLACE, "b = 0.01"),)),
)
#: The drawings the depth camera's radius covers whole: what stands up in the
#: hand (a cup, a candle).
WHOLE = ("cup", "candle")
#: THE PASS AS THE FORK'S WALK BEGINS (#762; recorded off the base's draw 12
#: over the tour, SimNotes): the expo counter's middle in the stance -- ahead,
#: left, m, and its turn, rad -- at the first scan carried after the fork's
#: hand-over, the robot 10.5 deg turned off square. There the front stop fired
#: on the counter's top edge, its nearest point 0.159 m from the fork's grip,
#: and backed the robot off it before the walk turned.
PASS_AT = (0.757, -0.048, math.radians(10.5))


def _turn(fingers, jaws) -> np.ndarray:
  """A reach's grip turn off its fingers and jaws, as `steps._grip_turn`
  makes it: x the jaws across the fingers, z against the fingers."""
  a = np.asarray(fingers, dtype=float) / np.linalg.norm(fingers)
  c = np.asarray(jaws, dtype=float)
  x = c - (a @ c) * a
  x /= np.linalg.norm(x)
  return np.column_stack([x, np.cross(-a, x), -a])


def _surface(model, data, body: int, step: float = 0.004) -> np.ndarray:
  """Points over every face of a body's geoms (boxes and cylinders), world."""
  out = []
  for g in np.flatnonzero(model.geom_bodyid == body):
    pos, mat, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
    local = []
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
      for ax in range(3):
        u, v = [k for k in range(3) if k != ax]
        for su in np.linspace(-1, 1, max(2, int(2 * size[u] / step) + 1)):
          for sv in np.linspace(-1, 1, max(2, int(2 * size[v] / step) + 1)):
            for sa in (-1.0, 1.0):
              p = np.zeros(3)
              p[ax], p[u], p[v] = sa * size[ax], su * size[u], sv * size[v]
              local.append(p)
    elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
      r, h = size[0], size[1]
      for t in np.linspace(0, 2 * math.pi, max(12, int(2 * math.pi * r / step)), endpoint=False):
        for z in np.linspace(-h, h, max(2, int(2 * h / step) + 1)):
          local.append((r * math.cos(t), r * math.sin(t), z))
        for rr in np.linspace(0.0, r, max(2, int(r / step) + 1)):
          local += [(rr * math.cos(t), rr * math.sin(t), -h), (rr * math.cos(t), rr * math.sin(t), h)]
    out.append(pos + np.asarray(local) @ mat.T)
  return np.concatenate(out)


@pytest.fixture(scope="module")
def carry():
  """The bare body with every drawing as a body a test poses, the expo
  counter too, and a pose: `pose(side, grip, fingers, jaws, drawing,
  counter=None)` puts the body at the stand -- both arms as the stand has
  them, whatever a test before posed -- the arm at its carry pose for a hand
  turned so (`Reach.carry_pose`) and the drawing in the hand as gripped, the
  others far off, and the counter where `counter` (ahead, left, turn) puts
  it in the stance, else far off -- returning the body, the true grip site
  and the drawing's body id."""
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  names = ("plate", "cup", "fork", "cube")
  for n in names:
    _piece(n)(spec, f"held_{n}")
  cyl = spec.worldbody.add_body(name="held_candle", mocap=True)
  cyl.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.035, 0.05, 0.0], pos=[0.0, 0.0, 0.05])
  # ...the expo counter's box, its geom on its body's middle (`cull.static_inertials`)
  counter = spec.worldbody.add_body(name="expo_counter", mocap=True)
  counter.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(home.EXPO_HALF))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  away = (20.0, 20.0, -5.0)

  def pose(side, grip, fingers, jaws, drawing, counter=None):
    m = body.mission
    mujoco.mj_resetDataKeyframe(model, data, 0)
    turn = _turn(fingers, jaws)
    m.reach.cmd[side] = HandGoal(np.zeros(3), turn, 1.0)
    q = m.reach.carry_pose(side)
    for j, v in zip(hm.ARM, q):
      data.qpos[model.jnt_qposadr[model.joint(f"{side}_{j}").id]] = v
    mujoco.mj_forward(model, data)
    site = model.site(f"{side}_grip").id
    at, rot = data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()
    for n in (*names, "candle"):
      data.mocap_pos[model.body_mocapid[model.body(f"held_{n}").id]] = away
    b = model.body(f"held_{drawing}").id
    mid = model.body_mocapid[b]
    data.mocap_pos[mid] = at + rot @ (turn.T @ -np.asarray(grip, dtype=float))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rot @ turn.T).reshape(-1))
    data.mocap_quat[mid] = quat
    cid = model.body_mocapid[model.body("expo_counter").id]
    data.mocap_pos[cid], data.mocap_quat[cid] = away, (1.0, 0.0, 0.0, 0.0)
    if counter is not None:
      ahead, left, yaw = counter
      data.mocap_pos[cid] = (ahead, left, home.EXPO_HALF[2])
      data.mocap_quat[cid] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    mujoco.mj_forward(model, data)
    # ...and the robot's model of itself posed as its senses would say it
    qs = np.array([data.qpos[model.jnt_qposadr[model.joint(n).id]] for n in hm.JOINT_NAMES])
    m.odo.sm.pose(qs, np.array([0.0, 0.0, -1.0]))
    return body, at, b

  yield pose
  body.close()


def _carried(what):
  return next(c for c in (*CARRIED, CANDLE) if c[0] == what)


def _needs(carry, what):
  """What a piece needs of each radius, m: (its farthest point from the true
  grip the depth camera sees, GRIP_OFF_M and two sigmas of its noise there;
  its farthest point the LIDAR's rings reach and the range noise's sigma
  there -- None where no ring reaches it at the carry pose)."""
  _, drawing, side, grip, fingers, jaws, *_ = _carried(what)
  body, at, b = carry(side, grip, fingers, jaws, drawing)
  model, data = body.model, body.data
  pts = _surface(model, data, b)
  far = np.linalg.norm(pts - at, axis=1)
  # the depth camera looks down on all of it, its noise along its axis
  dep = body.mission.depth
  cam, cmat = data.cam_xpos[dep.cam_id], data.cam_xmat[dep.cam_id].reshape(3, 3)
  z = -((pts - cam) @ cmat)[:, 2]
  depth = float((far + GRIP_OFF_M + 2 * pd.NOISE_K * z ** 2).max())
  # the LIDAR's rings reach what stands at or over its steepest
  lid = body.mission.lidar
  spos, smat = data.site_xpos[lid.site_id], data.site_xmat[lid.site_id].reshape(3, 3)
  v = (pts - spos) @ smat
  seen = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])) >= lid.elevations.min() - 1e-9
  if not seen.any():
    return depth, None
  k = int(np.argmax(far[seen] + pl._sigma(np.linalg.norm(v[seen], axis=1))))
  return depth, (float(far[seen][k]), float(pl._sigma(np.linalg.norm(v[seen][k]))))


def _fires(body, side, scans: int, radius: float, off: float = 0.0, piece: int | None = None) -> int:
  """Scans of the LIDAR's, its noise drawn (seeded), whose band laid flat with
  the grip held fires the front stop, at `radius` -- the grip as the robot's
  model of itself places it, or `off` m further from the middle of `piece`'s
  geoms: the worst way the model may be off."""
  m, model, data = body.mission, body.model, body.data
  m.lidar.rng = np.random.default_rng(762)
  root = model.body(hm.ROBOT_ROOT).id
  rot = data.xmat[root].reshape(3, 3)
  yaw = math.atan2(rot[1, 0], rot[0, 0])
  c, s = math.cos(yaw), math.sin(yaw)
  level = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot
  joints = {j: 0.0 for j in hm.WAIST}
  real = m.reach.grippers[side].held
  m.reach.grippers[side].held = lambda: True
  try:
    centres, _ = m._held_drop()
    assert len(centres) == 1, "one grip held: the worst way is its own piece's"
    if off:
      mid = data.geom_xpos[np.flatnonzero(model.geom_bodyid == piece)].mean(axis=0) - data.xpos[root]
      mid = np.array([c * mid[0] + s * mid[1], -s * mid[0] + c * mid[1], data.xpos[root][2] + mid[2]])
      away = centres[0] - mid
      centres = centres + off * away / np.linalg.norm(away)
    n = 0
    for _ in range(scans):
      scan = m.lidar.band(data, level, float(data.xpos[root][2]), joints=joints,
                          drop=(centres, radius))
      n += bool(len(m.in_front(scan.angles, scan.ranges)))
    return n
  finally:
    m.reach.grippers[side].held = real


# ---- 1: the LIDAR's radius between two measures ---------------------------------


def test_the_lidars_radius_covers_a_carried_candles_returns_at_one_sigma(carry):
  """#762: #736's candle, carried, stands in the LIDAR's steepest ring, its
  near side and top 0.5-0.6 m out, a range read 3.9 cm a sigma long or short:
  its farthest point the rings reach, the 3 cm the model may place the grip
  off the true one and one sigma, 0.154 m. At 0.12 its own returns fired the
  front stop on a fifth of the scans carried and every carried walk crawled
  (123-273 s for 38-46); from 0.16, all four arrived. Two sigmas would need
  0.193, past the pass's edge (below)."""
  _, lidar = _needs(carry, CANDLE[0])
  far, sigma = lidar
  assert far + GRIP_OFF_M + sigma <= hmis.HELD_DROP_M, (
    f"a candle's returns need {far + GRIP_OFF_M + sigma:.3f} m of the LIDAR's {hmis.HELD_DROP_M}")


def test_the_lidars_radius_keeps_the_pass_edge_beside_a_carried_fork_in_view(carry):
  """#762: as the fork's walk begins at the pass (`PASS_AT`), the counter's
  top edge runs across the front stop's box, its nearest point 0.16 m from
  the fork's grip and the rest of it further: there the front stop backs the
  robot off it before the walk turns. A radius hides the edge's returns
  within it, on more of the scans the wider it is: here, of 40 scans, the
  front stop fires on 39 at 0.12, 31 at 0.16, 16 at 0.20 and 6 at 0.25.
  Hidden, the walk turned at the pass and the fork's head struck its face --
  12 and 14 times in 32 places at 0.25 and 0.22, against 6 at 0.12 and 6-8 at
  0.16."""
  _, drawing, side, grip, fingers, jaws, _ = _carried(CARRIED[2][0])
  body, at, _ = carry(side, grip, fingers, jaws, drawing, counter=PASS_AT)
  data, model = body.data, body.model
  g = np.flatnonzero(model.geom_bodyid == model.body("expo_counter").id)[0]
  local = (at - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
  hx, hy, hz = home.EXPO_HALF
  edge = math.hypot(math.hypot(local[0] + hx, local[2] - hz), max(abs(local[1]) - hy, 0.0))
  assert 0.15 < edge < 0.17, f"the scene: the pass's edge {edge:.3f} m from the grip"
  n = 40
  assert _fires(body, side, n, hmis.HELD_DROP_M) >= n // 2, "the pass's edge hidden from the front stop"
  assert _fires(body, side, n, WIDE_M) <= n // 5, "the premise: at 0.25 the edge is hidden"


# ---- 2: a carried candle rarely stops the walk ----------------------------------


def test_a_carried_candle_rarely_stops_the_walk_at_the_lidars_radius(carry):
  """#762: the candle-sized piece held at the carry pose, the LIDAR cast at it
  with its noise: at 0.12 its own returns fire the front stop on a fifth of
  the scans with the grip as the model places it, and on two in five with
  the model 3 cm off the worst way; at the LIDAR's radius, on one in a
  hundred and one in sixteen (400 scans of the test's own cast) -- the
  carried candle's walks of #762's flights fired it 9-12 times each, and
  arrived in 38-46 s."""
  _, drawing, side, grip, fingers, jaws = CANDLE
  body, _, b = carry(side, grip, fingers, jaws, drawing)
  n = 40
  assert _fires(body, side, n, OLD_M) >= n // 10, "the premise: at 0.12 its own returns stop it"
  assert _fires(body, side, n, OLD_M, GRIP_OFF_M, b) >= n // 4, "the premise, the grip 3 cm off"
  assert _fires(body, side, n, hmis.HELD_DROP_M) <= n // 20
  assert _fires(body, side, n, hmis.HELD_DROP_M, GRIP_OFF_M, b) <= n // 10


# ---- 3: no carried piece stops the walk, and the grips are the references' ----


@pytest.mark.parametrize("what", [c[0] for c in CARRIED])
def test_no_carried_piece_stops_the_walk_at_the_lidars_radius(carry, what):
  """Every piece a job carries, its grip 3 cm off the worst way, fires the
  front stop on at most one scan in ten at the LIDAR's radius: a cup's own
  returns lie within 0.05 m of its grip, a fork's beside the front stop's box,
  and a plate and the cube stand under the steepest ring. A reference that
  grips a piece where its own returns would stop the walk fails here."""
  _, drawing, side, grip, fingers, jaws, _ = _carried(what)
  body, _, b = carry(side, grip, fingers, jaws, drawing)
  n = 20
  assert _fires(body, side, n, hmis.HELD_DROP_M, GRIP_OFF_M, b) <= n // 10


@pytest.mark.parametrize("what,refs", [(c[0], c[6]) for c in CARRIED], ids=[c[0] for c in CARRIED])
def test_the_grips_are_the_references_own(what, refs):
  """A reference that moves its grip -- further along a fork, higher on a
  cup -- can carry a piece past the radius: it fails here first, and its
  grip is read again in `CARRIED`."""
  for source, words in refs:
    text = source if isinstance(source, str) else "\n".join(source)
    assert words in text, f"{what}: its reference no longer says {words!r}"


# ---- 4: the depth camera's radius ----------------------------------------------


def test_the_depth_cameras_radius_covers_a_cup_or_a_candle_whole_and_no_more(carry):
  """The depth camera's noise is a few mm (`depth.NOISE_K` z^2), and its
  frame is what a reach's swing, its trunk and a step's knee room are judged
  against: what a radius hides near a held hand -- a counter's face beside a
  handle -- those judgments never see. #682's 0.12 covers what stands up in
  the hand whole, its farthest point, the 3 cm and two sigmas (a candle
  0.117, a cup 0.115), to the cm; a carried plate's rim (0.20) and a fork's
  head (0.17) stay in its frame, as they did -- at 0.20 the pass laid no
  better (#762: 46 of 64 against 50)."""
  needs = [_needs(carry, c[0])[0] for c in (*CARRIED, CANDLE) if c[1] in WHOLE]
  assert max(needs) <= hmis.HELD_DEPTH_M
  assert hmis.HELD_DEPTH_M == math.ceil(max(needs) * 100 - 1e-6) / 100, (
    f"the depth camera's radius {hmis.HELD_DEPTH_M} is wider than a cup or a candle needs, "
    f"{max(needs):.3f}")


# ---- 5: each sensor at its own radius --------------------------------------------


def test_the_band_drops_at_the_lidars_radius_and_the_depth_frame_at_its_own(carry, monkeypatch):
  """The LIDAR's noise is centimetres and the depth camera's millimetres: the
  band (the map, the front stop, what a step's room reads at the head's
  height) leaves out what lies within `HELD_DROP_M` of a held grip, the
  depth camera's frame (its low layer, its knees' and arms' memories) what
  lies within `HELD_DEPTH_M` -- the narrower, so a counter's face near a
  held handle stays in what a reach's swing is judged against."""
  _, drawing, side, grip, fingers, jaws = CANDLE
  body, _, _ = carry(side, grip, fingers, jaws, drawing)
  m = body.mission
  monkeypatch.setattr(m.reach.grippers["left"], "held", lambda: True)
  drops = {}
  monkeypatch.setattr(m.lidar, "band", lambda *a, **kw: drops.setdefault("band", kw["drop"]))
  m._scan()
  real = hmis._held_out

  def held_out(points, drop):
    drops["depth"] = drop
    return real(points, drop)
  monkeypatch.setattr(hmis, "_held_out", held_out)
  frame = m.latest_depth()
  assert frame is not None, "the premise: the bare body's depth camera gives a frame"
  m._frame_used_t, m._frame_level = -math.inf, True
  monkeypatch.setattr(m, "latest_depth", lambda: frame)
  monkeypatch.setattr(m, "level", lambda: True)
  monkeypatch.setattr(m, "_place_step", lambda: None)
  m._look_step()
  assert drops["band"][1] == hmis.HELD_DROP_M
  assert drops["depth"][1] == hmis.HELD_DEPTH_M
  assert hmis.HELD_DEPTH_M < hmis.HELD_DROP_M
  assert np.allclose(drops["band"][0], drops["depth"][0]), "one grip, placed once"
