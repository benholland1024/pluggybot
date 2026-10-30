"""The humanoid's charging station (issue #559, stage 2; SimNotes, "The
humanoid's charging station"): a box on a wall holding a plug on a cable,
which a humanoid plugs into the port on its own trunk (`humanoid.model`)
and charges standing.

Its parts are real (Parts.md, "The humanoid body"): Rosenberger's RoPD, the
e-bike charging connector -- magnetic, self-locating, 60 V, and pulled off
past 20 N -- whose plug rides a Tecna 9300 tool balancer, a spring that
draws it home into a cup on the station's face whenever nothing holds it
(Ben, 2026-10-30): an unplug or a breakaway leaves the station as it was,
the plug nose-out toward the robot. The charger is a Mean Well NPB-750-48 at
10 A, switched on only once the connector's signal pins are mated (the RoPD
is not hot-pluggable).

Two CATCHES are the activity, each a weld switched on where the plug is and
drawn to its seat (`PULL_IN_S`), and off past its force: the PORT, the
magnet on the robot's trunk (`BREAK_N`), and the CUP, the balancer's hold
on the plug's back (`PARK_N`). ⚠ THE CUP IS A CATCH, NEVER THE CUP'S SHAPE:
held in a socket by contact alone, the plug rattled (0.1 m/s, 6.7 deg), and
after a breakaway it swung on its rope under the box and lodged inside it;
the funnel that turns the plug's back into the socket is the catch's to
model, from anywhere in its mouth (`CUP_MOUTH_M`). The plug meets the
robots, the house and the station's box -- the rest of the station is
drawn, never touched.

THE STATION'S FRAME is the dock's (`legs.dock`): its origin on the floor
under the cup on the station's face, +x INTO the wall -- the way a robot at
the station faces -- y to the robot's left. The box stands between its face
and the wall (`BOX_DEPTH_M`).
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.activity.base import Activity
from pluggybot.rack.tags import STATION_TAG_IDS, STATION_TAG_SIZE, plate_half_extent

# ---- the parts' numbers ----------------------------------------------------------

#: The connector's hold, N: the RoPD's "disengagement force min. 20 N", every
#: sheet (Rosenberger, C001-04/C003-03 and their panel plugs); no maximum is
#: published. A pull off its face or across it past this lets go -- a magnet's
#: shroud levers off no harder than it pulls off -- and a push into it is
#: held by the face. ⚠ A robot unplugging itself pulls past it, and standing
#: it pulls about 35 N (SimNotes, "The humanoid's pull"): a part at its
#: unpublished maximum may hold harder.
BREAK_N = 20.0
#: How near the port takes the plug, m and deg: the gap off the connector's
#: face, the plug's offset across it and its tilt -- anywhere in the mouth
#: of the funnel printed round it (`humanoid.parts.PORT_FUNNEL`: 50 mm
#: across, 12 mm deep, round the 24.5 mm jack), which guides it onto the
#: connector. ⚠ THE FUNNEL IS THE CATCH'S, as the cup's is: its walls are
#: drawn, never touched. Pushed to the body from where the hand believed the
#: port to be, plugs landed 4-11 mm across it (`station_spike.py`).
CAPTURE_GAP_M, CAPTURE_LATERAL_M, CAPTURE_TILT_DEG = 0.012, 0.012, 10.0
#: The catch draws the plug from where it took it to its seat over this, s:
#: switched on as a weld at the seat, the plug was thrown in a step.
PULL_IN_S = 0.1
#: The plug clear of the connector again before it takes it again, m: a
#: breakaway would otherwise catch on the next step.
CLEAR_M = 0.012

#: The balancer's pull, N: the Tecna 9300's least, a 0.2 kg load (its range
#: 0.2-0.5 kg, catalogue DCB0000002 rev.04), set as light as it goes: the
#: plug and its cable's first length weigh 0.5 N.
BALANCER_N = 0.2 * 9.81
#: ...and the damping it reels in against, N s/m: UNPUBLISHED (a balancer's
#: drum runs against its own friction); at 8 the plug comes home at 0.25 m/s.
BALANCER_DAMPING = 8.0
#: Its rope's stop, m from the cup: the robot charging stands its port
#: 0.25 m off the cup, and walked or fallen this far the stop pulls the plug
#: off it (the balancer's own 1.6 m stroke would let a fallen robot lie
#: plugged in).
ROPE_STOP_M = 0.6
#: The cup lets the plug go to a pull straight out past this, N: the
#: balancer's 2 N and the funnel's hold, UNPUBLISHED (assumed); across it,
#: its walls hold.
PARK_N = 4.0
#: The cup's mouth, m: the plug's back within this of its seat is drawn in.
CUP_MOUTH_M = 0.03
#: ...and over this, s: turned from however it hangs.
PARK_IN_S = 0.3
#: The plug's drag, N s/m (and N m s/rad): its cable hanging in a loop beside
#: the rope. UNPUBLISHED: without it the plug swung on the rope at 2-3 m/s
#: after a breakaway and never reached the cup.
CABLE_DAMPING = 0.5

#: The charger's current, A: the NPB-750-48 set to 10 A of its 11.3 A (and
#: its CV to 50.4 V: its default boost, 57.6 V, is past the pack's), and
#: what it gives the pack, W, at the pack's nominal 43.2 V.
CHARGE_A = 10.0
CHARGE_W = CHARGE_A * 43.2

# ---- the plug ----------------------------------------------------------------------

#: The plug, its frame at its mating face, +x out of it: the RoPD cable
#: jack (C003-03: 20 g, 24.5 mm across, 43 mm long) and its cable gland
#: behind it, 70 mm in all; with its cable's first 0.3 m (Rosenberger's
#: C003 cable, 101 g/m), 50 g.
PLUG_LEN_M, PLUG_R_M, PLUG_KG = 0.070, 0.01225, 0.050
JACK_LEN_M = 0.043
#: ...its inertia, kg m^2 (about its axis, across it): across, its cable's
#: first 0.3 m hanging behind it (30 g on a 0.15 m lever); about its axis,
#: REGULARISED -- the bare jack's 2e-6, pinched between a gripper's pads,
#: threw the plug's rotation off in a step, and 5e-5, spun loose, a few
#: seconds on (`implicitfast` drops the spin's derivative).
PLUG_INERTIA = (2e-4, 9e-4)
#: Where a hand grips it, m behind its face: the jack's back end, so the
#: Dex1-1's housing, 67 mm across the plug's axis, stands behind its face
#: -- gripped at the jack's middle, the housing met the robot's belly with
#: the plug 17 mm off its port.
GRIP_BACK_M = 0.040

# ---- the station ---------------------------------------------------------------------

#: The cup's height, m: the robot's port standing (`humanoid.model`'s
#: `port_z` over the stand's pelvis), so the plug is carried level.
CUP_Z = 0.90
#: The box between the station's face and the wall, m: deep enough for the
#: balancer (Tecna's 9300) and the charger is not in it (on the wall beside).
BOX_DEPTH_M = 0.10
#: The cup stands this far proud of the face on a boss, m: a hand's housing
#: round the parked plug's grip clears the face by 30 mm.
BOSS_M = 0.030
BOX_HALF_Y, BOX_HALF_Z = 0.15, 0.20
#: Its tags, a pair either side of the cup at its height (the dock's 60 mm):
#: their baseline gives its facing (#88), and from the seat both are in the
#: colour camera's view (54 deg below level, 32 off its axis at most).
TAG_Y = 0.09
#: Where the robot stands to plug in, in the station's frame, m: its stance
#: 0.325 m off the face, 7.5 cm to the cup's left so its port, right of its
#: middle (`humanoid.model`'s `port_y`), is on the cup's axis: the parked
#: plug's grip 0.28 m ahead of it, 0.18 m of carry to its port
#: (SimNotes, "Where the port goes").
SEAT_X, SEAT_Y = -0.325, 0.075
#: ...and where its walk ends, m: the tags in view and room to line up.
STANDOFF_X = -0.9


def _f(v: float) -> str:
  return f"{v:.6g}"


def tag_layout() -> dict[int, tuple[float, float, float]]:
  """Each tag's centre in the station's frame, by id: its drawing, what a
  robot knows of the station before it sees it."""
  return {STATION_TAG_IDS[0]: (0.0, TAG_Y, CUP_Z), STATION_TAG_IDS[1]: (0.0, -TAG_Y, CUP_Z)}


def station_xml(name: str = "station") -> str:
  """The station as a document to attach (`attach_station`): its box, the
  cup and the tags drawn on its face, the plug on its balancer's rope, the
  cup's catch. The robots' catches are added by `attach_station`."""
  half = plate_half_extent(STATION_TAG_SIZE)
  g = [f'<geom name="{name}_box" type="box" size="{_f(BOX_DEPTH_M / 2)} {_f(BOX_HALF_Y)} '
       f'{_f(BOX_HALF_Z)}" pos="{_f(BOX_DEPTH_M / 2)} 0 {_f(CUP_Z)}" rgba="0.30 0.31 0.34 1"/>',
       f'<geom name="{name}_boss" type="cylinder" size="0.02 {_f(BOSS_M / 2)}" '
       f'pos="{_f(-BOSS_M / 2)} 0 {_f(CUP_Z)}" zaxis="1 0 0" contype="0" conaffinity="1" '
       f'rgba="0.55 0.56 0.6 1"/>',
       f'<site name="{name}_rope" pos="{_f(-BOSS_M - 0.003)} 0 {_f(CUP_Z)}" size="0.004"/>',
       f'<site name="{name}_seat" pos="{_f(-BOSS_M - 0.003)} 0 {_f(CUP_Z)}" size="0.004" '
       f'zaxis="-1 0 0"/>']
  for tag_id, (x, y, z) in tag_layout().items():
    g.append(f'<geom name="{name}_tag{tag_id}" type="box" size="0.001 {_f(half)} {_f(half)}" '
             f'pos="{_f(x - 0.001)} {_f(y)} {_f(z)}" contype="0" conaffinity="0" '
             f'material="tagmat{tag_id}"/>')
  # ...the plug parked: nose-out (-x), its back on the seat
  px = -BOSS_M - 0.003 - PLUG_LEN_M
  plug = (f'<body name="{name}_plug" pos="{_f(px)} 0 {_f(CUP_Z)}" quat="0 0 0 1">'
          f'<joint name="{name}_plug" type="free" damping="{_f(CABLE_DAMPING)}"/>'
          f'<inertial pos="{_f(-PLUG_LEN_M / 2)} 0 0" mass="{_f(PLUG_KG)}" '
          f'diaginertia="{_f(PLUG_INERTIA[0])} {_f(PLUG_INERTIA[1])} {_f(PLUG_INERTIA[1])}"/>'
          f'<geom name="{name}_jack" type="cylinder" size="{_f(PLUG_R_M)} {_f(JACK_LEN_M / 2)}" '
          f'pos="{_f(-JACK_LEN_M / 2)} 0 0" zaxis="1 0 0" '
          f'contype="2" conaffinity="1" priority="1" friction="1 0.005 0.0001" '
          f'solimp="0.99 0.999 0.0001" rgba="0.12 0.12 0.13 1"/>'
          f'<geom name="{name}_gland" type="cylinder" size="0.009 {_f((PLUG_LEN_M - JACK_LEN_M) / 2)}" '
          f'pos="{_f(-(PLUG_LEN_M + JACK_LEN_M) / 2)} 0 0" zaxis="1 0 0" '
          f'contype="2" conaffinity="1" rgba="0.08 0.08 0.08 1"/>'
          f'<site name="{name}_plug_face" pos="0 0 0" size="0.003"/>'
          f'<site name="{name}_plug_back" pos="{_f(-PLUG_LEN_M)} 0 0" size="0.003"/>'
          f'</body>')
  return (f'<mujoco><compiler angle="radian"/><worldbody>'
          f'<body name="{name}">' + "".join(g) + '</body>' + plug + '</worldbody>'
          f'<tendon><spatial name="{name}_rope" limited="true" range="0 {_f(ROPE_STOP_M)}" '
          f'solreflimit="0.01 1" rgba="0.8 0.8 0.8 1" width="0.002">'
          f'<site site="{name}_rope"/><site site="{name}_plug_back"/></spatial></tendon>'
          f'<actuator><general name="{name}_balancer" tendon="{name}_rope" gainprm="0" '
          f'biastype="affine" biasprm="{_f(-BALANCER_N)} 0 {_f(-BALANCER_DAMPING)}"/></actuator>'
          f'<equality><weld name="{name}_cup" body1="{name}" body2="{name}_plug" '
          f'solref="0.004 1"/></equality></mujoco>')


def attach_station(spec, pose, name: str = "station", roots=("pluggybot",)) -> None:
  """A station into a world's `spec` at `pose` (x, y, yaw rad: its frame,
  above), with a catch for each robot's port (`roots`, the robots' root
  bodies, attached first), and its tags' textures once a world."""
  import mujoco
  for tag_id in STATION_TAG_IDS:
    if spec.texture(f"tagtex{tag_id}") is None:
      spec.add_texture(name=f"tagtex{tag_id}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                       file=f"tags/tag{tag_id}.png")
      mat = spec.add_material(name=f"tagmat{tag_id}", specular=0.05, shininess=0.05,
                              reflectance=0.0)
      mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{tag_id}"
  x, y, yaw = pose
  frame = spec.worldbody.add_frame()
  frame.pos = [float(x), float(y), 0.0]
  frame.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  spec.attach(mujoco.MjSpec.from_string(station_xml(name)), prefix="", frame=frame)
  for root in roots:
    spec.add_equality(name=f"{name}_port_{root}", type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY, name1=root, name2=f"{name}_plug",
                      active=False, solref=[0.004, 1.0])


def to_world(pose, x: float, y: float) -> tuple[float, float]:
  """A point of the station's frame in the world's."""
  px, py, yaw = pose
  c, s = math.cos(yaw), math.sin(yaw)
  return px + c * x - s * y, py + s * x + c * y


# ---- the activity ---------------------------------------------------------------------


def _quat_of(rot: np.ndarray) -> np.ndarray:
  import mujoco
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.ascontiguousarray(rot, dtype=float).flatten())
  return q


class Station(Activity):
  """The station's catches (the module docstring), sensed every step: the
  CUP holding the plug's back, and each robot's PORT holding its face.
  `flags`: `mated` (the root whose port holds the plug, or ""), `parked`
  (the cup holds it) and `breaks` (how many times a port let it go past its
  hold)."""

  def __init__(self, model, data, name: str = "station") -> None:
    super().__init__(name)
    self.rebind(model, data)
    #: Per catch (eq id): (when it took the plug, from where, its turn), or
    #: None once seated; whether it will take the plug again; when.
    self._pull: dict[int, tuple | None] = {}
    self._armed = {eq: True for eq in self.catches}
    self.breaks = 0
    self.set(mated="", parked=bool(data.eq_active[self.cup]), breaks=0)

  def rebind(self, model, data) -> None:
    import mujoco
    n = self.name
    self.m = model
    self.plug = model.body(f"{n}_plug").id
    self.face = model.site(f"{n}_plug_face").id
    self.back = model.site(f"{n}_plug_back").id
    self.seat = model.site(f"{n}_seat").id
    self.cup = model.equality(f"{n}_cup").id
    #: Each robot's catch: its eq id -> (its root body, its port site).
    self.ports = {}
    prefix = f"{n}_port_"
    for e in range(model.neq):
      name = model.equality(e).name
      if name.startswith(prefix):
        root = name[len(prefix):]
        port = model.site(f"{root[:-len('pluggybot')]}charge_port").id
        self.ports[e] = (int(model.eq_obj1id[e]), port)
    self.catches = (self.cup, *self.ports)
    # The seats, each the plug's pose in the catch's body1's frame. The
    # port's: the plug's face on the port, its +x into the robot (the
    # port's z out of it); the cup's: its back on the seat, nose out.
    station = int(model.eq_obj1id[self.cup])
    self.body1 = {self.cup: station, **{e: b for e, (b, _) in self.ports.items()}}
    self._seat = {}
    for e, (root, port) in self.ports.items():
      r = np.zeros(9)
      mujoco.mju_quat2Mat(r, model.site_quat[port])
      out = r.reshape(3, 3)[:, 2]
      x = -out
      z = np.array([0.0, 0.0, 1.0])
      y = np.cross(z, x)
      self._seat[e] = (model.site_pos[port].copy(), _quat_of(np.column_stack([x, y, z])))
    sx = model.site_pos[self.seat].copy()
    self._seat[self.cup] = (sx + np.array([-PLUG_LEN_M, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    model.eq_data[self.cup, 0:3] = 0.0
    model.eq_data[self.cup, 3:6], model.eq_data[self.cup, 6:10] = self._seat[self.cup]
    # efc rows are found each step a catch holds; the arrays it reads
    self._eq_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)

  # ---- reading the physics ------------------------------------------------------

  def _rel_now(self, data, body1: int) -> tuple[np.ndarray, np.ndarray]:
    """The plug's pose in `body1`'s frame, now."""
    import mujoco
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, data.xquat[body1])
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, data.xpos[self.plug] - data.xpos[body1], inv)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, inv, data.xquat[self.plug])
    return pos, q

  def _load(self, data, eq: int) -> np.ndarray:
    """What pulls the plug off a catch's seat, N (world): its weld's force."""
    rows = np.flatnonzero((data.efc_type[:data.nefc] == self._eq_type)
                          & (data.efc_id[:data.nefc] == eq))
    return data.efc_force[rows[:3]].copy() if len(rows) >= 3 else np.zeros(3)

  def _off(self, f: np.ndarray, out: np.ndarray) -> float:
    """A load's pull off a face whose outward normal is `out`: along it and
    across it -- a push into the face is the face's."""
    along = float(f @ out)
    across = float(np.linalg.norm(f - along * out))
    return math.hypot(max(along, 0.0), across)

  def _port_offset(self, data, port: int) -> tuple[float, float, float]:
    """The plug's face against a port: its gap off the port's face, its
    offset across it (m) and its tilt (deg)."""
    out = data.site_xmat[port].reshape(3, 3)[:, 2]
    d = data.site_xpos[self.face] - data.site_xpos[port]
    gap = float(d @ out)
    lateral = float(np.linalg.norm(d - gap * out))
    axis = data.xmat[self.plug].reshape(3, 3)[:, 0]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-axis @ out)))))
    return gap, lateral, tilt

  # ---- the catches -----------------------------------------------------------------

  def _take(self, data, eq: int) -> None:
    pos, q = self._rel_now(data, self.body1[eq])
    self.m.eq_data[eq, 0:3] = 0.0
    self.m.eq_data[eq, 3:6], self.m.eq_data[eq, 6:10] = pos, q
    self._pull[eq] = (float(data.time), pos, q)
    data.eq_active[eq] = 1

  def _draw_in(self, data, eq: int) -> None:
    import mujoco
    pull = self._pull.get(eq)
    if pull is None:
      return
    t0, p0, q0 = pull
    span = PARK_IN_S if eq == self.cup else PULL_IN_S
    a = min(1.0, (float(data.time) - t0) / span)
    p1, q1 = self._seat[eq]
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))  # q1 q0^-1
    angle = 2.0 * math.atan2(float(np.linalg.norm(q[1:])), float(q[0]))
    if angle > math.pi:
      q, angle = -q, 2 * math.pi - angle
    axis = q[1:] / max(float(np.linalg.norm(q[1:])), 1e-12)
    step = np.concatenate(([math.cos(a * angle / 2)], math.sin(a * angle / 2) * axis))
    qa = np.zeros(4)
    mujoco.mju_mulQuat(qa, step, q0)
    self.m.eq_data[eq, 3:6] = p0 + (p1 - p0) * a
    self.m.eq_data[eq, 6:10] = qa / np.linalg.norm(qa)
    if a >= 1.0:
      self._pull[eq] = None

  def _let_go(self, data, eq: int) -> None:
    data.eq_active[eq] = 0
    self._pull.pop(eq, None)
    self._armed[eq] = False

  def sense(self, model, data) -> None:
    mated = ""
    for eq, (root, port) in self.ports.items():
      out = data.site_xmat[port].reshape(3, 3)[:, 2]
      if data.eq_active[eq]:
        self._draw_in(data, eq)
        # ...judged once seated: a hand still on the plug fights its draw-in
        if self._pull.get(eq) is None and self._off(self._load(data, eq), out) >= BREAK_N:
          self._let_go(data, eq)
          self.breaks += 1
        else:
          mated = model.body(root).name
          continue
      gap, lateral, tilt = self._port_offset(data, port)
      if not self._armed[eq]:
        self._armed[eq] = gap > CLEAR_M or lateral > CLEAR_M
      elif (not mated and not data.eq_active[self.cup]
            and -0.002 <= gap <= CAPTURE_GAP_M and lateral <= CAPTURE_LATERAL_M
            and tilt <= CAPTURE_TILT_DEG):
        self._take(data, eq)
        mated = model.body(root).name
    cup_out = -data.site_xmat[self.seat].reshape(3, 3)[:, 2]     # into the station
    if data.eq_active[self.cup]:
      self._draw_in(data, self.cup)
      # ...only to a pull straight out of it: the cup's walls take a load
      # across it (the jaws closing round the parked plug let it go at 4 N
      # and it slid 15 mm in them)
      if (self._pull.get(self.cup) is None
          and float(self._load(data, self.cup) @ -cup_out) >= PARK_N):
        self._let_go(data, self.cup)
    else:
      dist = float(np.linalg.norm(data.site_xpos[self.back] - data.site_xpos[self.seat]))
      if not self._armed[self.cup]:
        self._armed[self.cup] = dist > CUP_MOUTH_M
      elif not mated and dist <= CUP_MOUTH_M:
        self._take(data, self.cup)
    self.set(mated=mated, parked=bool(data.eq_active[self.cup]), breaks=self.breaks)

  # ---- a restart -----------------------------------------------------------------

  def kept_state(self) -> dict:
    """The flags, and each catch's seat as it stands, its draw-in and
    whether it is armed: `eq_active` is the continuation's, by name, but a
    catch's pose is the MODEL's, which a restart builds fresh."""
    m = self.m
    return {"flags": dict(self.flags), "breaks": self.breaks,
            "catches": {m.equality(e).name: {
              "rel": [float(v) for v in m.eq_data[e, 3:10]],
              "armed": bool(self._armed[e]),
              "pull": None if self._pull.get(e) is None else
              [self._pull[e][0], [float(v) for v in self._pull[e][1]],
               [float(v) for v in self._pull[e][2]]]}
              for e in self.catches}}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    self.breaks = int(state.get("breaks", 0))
    for name, kept in (state.get("catches") or {}).items():
      try:
        e = self.m.equality(name).id
      except KeyError:
        continue
      self.m.eq_data[e, 3:10] = kept["rel"]
      self._armed[e] = bool(kept["armed"])
      pull = kept.get("pull")
      self._pull[e] = None if pull is None else (float(pull[0]), np.array(pull[1]),
                                                 np.array(pull[2]))
