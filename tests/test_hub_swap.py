"""Guards for the fork robot + hub world integration (milestone 8)."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.rack.coupling import HUB_STATION_YS, TOOL_HALF_X
from pluggybot.rack.swap import HubSwap

# Adjacent-link clearance for the FORK, the same geometric discipline as
# test_arm.py: weld/parent filtering silences these contacts, so overlap is
# asserted from geometry through the whole lift+arm envelope.
FORK_GEOMS = ("fork_bridge", "fork_prong_l", "fork_prong_r",
              "fork_vl_a", "fork_vl_b", "fork_vr_a", "fork_vr_b",
              # The lean-pad hangs lower than any of the above, so it is the
              # binding geom for the parked envelope -- it MUST be swept.
              "lean_pad", "lean_pad_post")
CLEARANCE_REFS = ("head_mount", "mast", "lift_motor", "chassis", "battery")


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def _world_half_extents(model, data, gid):
  """Box half-extents projected onto world axes (exact for axis-aligned
  geoms, a tight conservative bound for the 45-degree V plates -- unlike a
  bounding sphere, which inflates the wide flat bridge into a 65 mm ball)."""
  size = model.geom(gid).size
  xmat = data.geom_xmat[gid].reshape(3, 3)
  return [sum(abs(float(xmat[i][j])) * float(size[j]) for j in range(3))
          for i in range(3)]


def _aabb_overlap(model, data, a, b):
  ga, gb = model.geom(a).id, model.geom(b).id
  pa, pb = data.geom_xpos[ga], data.geom_xpos[gb]
  ha, hb = _world_half_extents(model, data, ga), _world_half_extents(model, data, gb)
  return all(abs(float(pa[k]) - float(pb[k])) < ha[k] + hb[k] for k in range(3))


def test_fork_stows_below_the_scan_row(hub_model):
  d = mujoco.MjData(hub_model)
  d.qpos[0] = 2.0                     # far from the hub
  mujoco.mj_forward(hub_model, d)
  for _ in range(1000):
    mujoco.mj_step(hub_model, d)
  for g in FORK_GEOMS:
    gid = hub_model.geom(g).id
    top = float(d.geom_xpos[gid][2]) + _world_half_extents(hub_model, d, gid)[2]
    assert top < 0.18, f"{g} reaches the scan row while parked (top {top:.3f})"


def test_fork_transit_clears_the_robot(hub_model):
  d = mujoco.MjData(hub_model)
  d.qpos[0] = 2.0
  mujoco.mj_forward(hub_model, d)
  for _ in range(500):
    mujoco.mj_step(hub_model, d)
  lift = hub_model.actuator("lift").id
  arm = hub_model.actuator("arm").id
  hits = set()
  for step in range(3000):
    d.ctrl[lift] = 0.31 * step / 3000
    mujoco.mj_step(hub_model, d)
    if step % 20 == 0:
      hits |= {(a, b) for a in FORK_GEOMS for b in CLEARANCE_REFS
               if _aabb_overlap(hub_model, d, a, b)}
  for step in range(1500):
    d.ctrl[arm] = 0.20 * step / 1500
    mujoco.mj_step(hub_model, d)
    if step % 20 == 0:
      hits |= {(a, b) for a in FORK_GEOMS for b in CLEARANCE_REFS
               if _aabb_overlap(hub_model, d, a, b)}
  assert not hits, f"fork sweeps through the robot: {sorted(hits)}"


def test_robot_swap_cycle(hub_model):
  """The milestone-8 integration claim: the real base + lift + odometry runs
  the coupling verbs end to end -- pick the LCD module off the rack, carry it
  away, bring it back, hang it up. And the freestanding rack must not move:
  stability is measured, not assumed (it is a free body leaning on the wall
  through its braces; without them a sustained press scooted it 9.6 mm)."""
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap.pick()
  st = swap.module_state("module_lcd")
  assert st["on_fork"], f"pick failed: module at {st['pos']}"
  assert st["pos"][0] > 0.15, "module did not leave the rack"
  swap.put_back()
  st = swap.module_state("module_lcd")
  assert st["hung"], f"return failed: module at {st['pos']}"
  rack = data.xpos[hub_model.body("rack").id]
  assert abs(float(rack[0])) < 0.005 and abs(float(rack[1])) < 0.005, \
    "the rack moved during the swap"


PEN_LEN = 0.050            # a pen tip sticking out past the module face
TOOL_FORCE_N = 0.5         # a whiteboard marker wants 0.5-2 N


def _pen_local():
  return np.array([-(TOOL_HALF_X + PEN_LEN), 0.0, 0.0])


def _module_vs_fork(model, data, name="module_lcd"):
  """(lean angle deg, pen tip in the fork's frame). Lean is the module's
  rotation about the peg axis relative to the fork -- the axis a tool force
  rotates it around."""
  bid, fid = model.body(name).id, model.body("tool_fork").id
  p = np.array(data.xpos[bid], dtype=float)
  r = np.array(data.xmat[bid], dtype=float).reshape(3, 3)
  fp = np.array(data.xpos[fid], dtype=float)
  fr = np.array(data.xmat[fid], dtype=float).reshape(3, 3)
  lean = math.degrees(math.atan2(float(r[:, 2] @ fr[:, 0]),
                                 float(r[:, 2] @ fr[:, 2])))
  tip = fr.T @ ((p + r @ _pen_local()) - fp)
  return lean, tip


def test_carried_module_can_exert_tool_force(hub_model):
  """A gravity latch cannot push, so a motorised tool needs the lean-pad.

  Measured on the bare hang (SimNotes): the restoring arm (CoM 22 mm below
  the peg) and the moment arm of a horizontal tool force about that peg are
  the SAME 22 mm, so any tool force buys a proportional, large rotation --
  0.1 N gave 5.3 deg, and past 0.25 N it runs away because rotating grows
  the disturbing arm faster than sin(theta) grows the restoring one. At
  0.5 N the unsupported module flopped 51 deg onto the fork and the pen tip
  retreated 38 mm.

  With the lean-pad the tool force is reacted in compression at a ~38 mm
  lever instead, so the module barely moves.
  """
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap.pick()
  assert swap.module_state("module_lcd")["on_fork"], "pick failed"
  swap._run(3.0, 0.0)

  bid = hub_model.body("module_lcd").id
  lean0, tip0 = _module_vs_fork(hub_model, data)

  # Board pushes the pen back: a force along the fork's +x, applied at the
  # tip (so xfrc carries the r x F torque too).
  for _ in range(3000):
    p = np.array(data.xpos[bid], dtype=float)
    r = np.array(data.xmat[bid], dtype=float).reshape(3, 3)
    fr = np.array(data.xmat[hub_model.body("tool_fork").id],
                  dtype=float).reshape(3, 3)
    f_world = fr[:, 0] * TOOL_FORCE_N
    data.xfrc_applied[bid][:3] = f_world
    data.xfrc_applied[bid][3:] = np.cross((p + r @ _pen_local()) - p, f_world)
    mujoco.mj_step(hub_model, data)
  lean, tip = _module_vs_fork(hub_model, data)
  data.xfrc_applied[bid] = 0.0

  assert abs(lean - lean0) < 2.0, (
    f"{TOOL_FORCE_N} N of tool force rotated the module "
    f"{abs(lean - lean0):.1f} deg -- the lean-pad is not reacting it")
  assert abs(float(tip[0] - tip0[0])) < 0.003, (
    f"pen tip retreated {abs(float(tip[0] - tip0[0])) * 1000:.1f} mm under "
    f"{TOOL_FORCE_N} N -- a drawing tool cannot hold a line like that")
  assert swap.module_state("module_lcd")["on_fork"], "tool force unseated it"


def test_module_power_follows_the_coupling(hub_model):
  """The module electrical interface: the peg IS the connector.

  Measured (SimNotes): the peg already sits in four V-notch plates carrying
  0.43-0.49 N each of gravity preload, 4-9x what a lean-pad could supply, so
  the power contacts belong there and nowhere else. Split into two conductors
  around an insulated centre, the left and right V-pairs are the two poles.

  What this guards is the criterion, not the geometry: a module hanging on
  the rack is DEAD, a module properly seated on the fork is LIVE, and the
  transition happens on the lift -- the latching motion -- not on arrival.
  """
  from pluggybot.rack.coupling import module_power_contact, module_power_state
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap._run(1.0, 0.0)

  assert not module_power_contact(hub_model, data), \
    "a module hanging on the rack must not be powered"

  swap.pick()
  swap._run(1.0, 0.0)
  st = module_power_state(hub_model, data)
  assert st["powered"], f"picked module is not conducting: {st}"
  assert st["left"] and st["right"], f"only one pole seated: {st}"

  swap.put_back()
  swap._run(1.0, 0.0)
  assert swap.module_state("module_lcd")["hung"], "return failed"
  assert not module_power_contact(hub_model, data), \
    "a stowed module must go dead -- the robot is what powers its tools"


def test_module_power_survives_carrying(hub_model):
  """Continuity must hold through the maneuvering an errand involves.

  The milestone-7 lesson in its new costume: contacts need sustained preload
  or the circuit opens (there, the suspension relaxed and charging stopped).
  Here the preload is gravity through the peg, and the question is whether
  the module's swing -- measured at 10.3 deg peak-to-peak through a turn --
  ever breaks a pole loose. A motorised tool that browns out mid-stroke
  because the robot turned would be a miserable bug to find later.
  """
  from pluggybot.control import wheel_targets
  from pluggybot.rack.coupling import module_power_state
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap.pick()
  assert swap.module_state("module_lcd")["on_fork"], "pick failed"

  state = {"start": None, "prev": 0.0, "spans": []}

  def drive(seconds, v, w):
    tl, tr = wheel_targets(v, w)
    for _ in range(round(seconds / hub_model.opt.timestep)):
      swap._step_once(tl, tr)
      powered = module_power_state(hub_model, data)["powered"]
      t = float(data.time)
      if not powered and state["start"] is None:
        state["start"] = t
      elif powered and state["start"] is not None:
        state["spans"].append(state["prev"] - state["start"])
        state["start"] = None
      state["prev"] = t
    return state

  # Turn away from the rack first, then a net-zero haul: out, back, turn,
  # counter-turn. (Driving straight forward after a pick rams the hub.)
  drive(2.7, 0.0, 1.2)
  for seconds, v, w in ((2.0, 0.25, 0.0), (1.0, 0.0, 0.0), (2.0, -0.25, 0.0),
                        (2.0, 0.0, 1.2), (2.0, 0.0, -1.2)):
    total = drive(seconds, v, w)

  assert swap.module_state("module_lcd")["on_fork"], "dropped the tool"
  # Judged on the DURATION of the worst outage, not on a percentage of steps.
  # A percentage cannot distinguish a hundred microsecond blips from one long
  # outage, and it is the long one that decides whether the module's holding
  # capacitor carries it -- the same argument module_power.py already makes.
  # Measured with honest peg friction (mu 0.4, steel on printed V): 20 spans,
  # worst 178 ms, 289 ms total across 11.7 s of deliberately harsh driving.
  # That is a real hardware number: the cap must cover ~200 ms, not the 50 ms
  # the release transient suggested.
  worst = max(total["spans"]) if total["spans"] else 0.0
  assert worst < 0.25, (
    f"worst brown-out {worst * 1000:.0f} ms while driving -- beyond what a "
    f"module holding capacitor can reasonably carry")


def test_bay_tag_ids_pair_by_index(hub_model):
  """Bay <-> tag pairing must survive a third bay. This was a hardcoded
  two-bay equality check, which would have silently steered every bay-C swap
  onto bay B's marker -- the exact class of bug real AprilTags were adopted
  to make impossible."""
  from pluggybot.rack.coupling import HUB_STATION_YS, bay_tag_id
  from pluggybot.rack.tags import BAY_TAG_IDS
  assert len(BAY_TAG_IDS) == len(HUB_STATION_YS), "a bay has no tag"
  assert len(set(BAY_TAG_IDS)) == len(BAY_TAG_IDS), "duplicate bay tag id"
  for i, y in enumerate(HUB_STATION_YS):
    assert bay_tag_id(y) == BAY_TAG_IDS[i]
    assert bay_tag_id(y + 0.01) == BAY_TAG_IDS[i], "nearest-bay lookup is tight"
  for i in range(len(HUB_STATION_YS)):
    gid = hub_model.geom(f"bay{chr(ord('a') + i)}_bay_tag").id
    assert gid >= 0


def test_pen_module_is_a_usable_tool(hub_model):
  """The drawing module: a tool that brings its own actuated axis.

  The robot owns x/yaw (base), z (lift) and reach (arm) -- and nothing owns
  LATERAL, because a differential drive cannot translate sideways. The pen
  carriage runs along the module's own peg axis and supplies exactly that,
  pairing with the lift to make an X-Y plotter. So the module has to survive
  being a module first: hang, be picked, and conduct.
  """
  from pluggybot.rack.coupling import (
    HUB_STATION_YS, PEN_TRAVEL, module_power_contact,
  )
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap._run(1.0, 0.0)
  assert not module_power_contact(hub_model, data, "module_pen")

  swap.pick()
  st = swap.module_state("module_pen")
  assert st["on_fork"], f"pen module pick failed: {st}"
  assert module_power_contact(hub_model, data, "module_pen"), \
    "the pen module is not powered through the coupling -- a motorised tool " \
    "with no power is cargo"

  # Drive the carriage end to end and check BOTH that it tracks and that the
  # tool stays seated. Seating is judged by the ELECTRICAL criterion, not by
  # module_state's on_fork: on_fork allows 3-4 cm of slop, and it cheerfully
  # reported True while the module had come out of the V-notches and yawed
  # 100 degrees. The poles are the honest answer, same as everywhere else.
  #
  # The setpoint is RAMPED. A stiff position servo handed a step command
  # delivers it as an impulse: an 80 mm jump threw the module off the fork,
  # while walking the same distance held seated throughout.
  act = hub_model.actuator("pen_carriage").id
  adr = hub_model.joint("pen_carriage_joint").qposadr[0]
  reached = []
  for target in (PEN_TRAVEL, -PEN_TRAVEL, 0.0):
    cur = float(data.ctrl[act])
    steps = max(int(abs(target - cur) / 0.03 / hub_model.opt.timestep), 1)
    for k in range(steps):
      data.ctrl[act] = cur + (target - cur) * (k + 1) / steps
      mujoco.mj_step(hub_model, data)
    swap._run(1.0, 0.0)
    reached.append(float(data.qpos[adr]))
    assert abs(float(data.qpos[adr]) - target) < 0.004, (
      f"carriage commanded to {target * 1000:.0f} mm reached "
      f"{float(data.qpos[adr]) * 1000:.1f} mm -- it is jammed on something")
    assert module_power_contact(hub_model, data, "module_pen"), (
      f"the carriage's own motion unseated the module at "
      f"{target * 1000:.0f} mm")
  assert max(reached) - min(reached) > 0.10, "carriage barely moved"


def test_pen_carriage_sweep_clears_the_robot(hub_model):
  """Geometric clearance for the carriage's full stroke, held on the fork.

  Same discipline as test_fork_transit_clears_the_robot and for the same
  reason: weld/parent filtering hides some of these contacts, and the pen
  module is the first part that MOVES while carried, so its envelope has
  never been swept before.
  """
  from pluggybot.rack.coupling import HUB_STATION_YS, PEN_TRAVEL
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  assert swap.module_state("module_pen")["on_fork"], "pick failed"

  pen_geoms = ("module_pen_block", "module_pen_shaft")
  # The module's OWN frame belongs in this list. The first version checked
  # the pen only against the robot, and missed that the carriage and shaft
  # were buried inside the module's body plate -- MuJoCo filters parent-child
  # contacts but NOT grandparent, so the quill fought the plate and the
  # carriage jammed at +21.9 mm partway through every figure.
  refs = CLEARANCE_REFS + ("lean_pad", "lean_pad_post", "fork_bridge",
                           "fork_prong_l", "fork_prong_r",
                           "module_pen_body", "module_pen_rail")
  act = hub_model.actuator("pen_carriage").id
  hits = set()
  for step in range(2400):
    data.ctrl[act] = PEN_TRAVEL * math.sin(2 * math.pi * step / 2400)
    mujoco.mj_step(hub_model, data)
    if step % 20 == 0:
      hits |= {(a, b) for a in pen_geoms for b in refs
               if _aabb_overlap(hub_model, data, a, b)}
  # The block wraps the rail it rides on -- that overlap IS the bearing, and
  # MuJoCo filters the pair anyway (parent-child). Everything else must clear,
  # including shaft-vs-rail, which is the grandparent pair that does collide.
  hits -= {("module_pen_block", "module_pen_rail")}
  assert not hits, f"pen carriage sweeps through something: {sorted(hits)}"


def _module_geoms(model, name):
  """Every colliding geom of a module INCLUDING its sub-bodies. The pen's
  rail is on the module plate but its carriage and quill are child bodies,
  and it was a child-body geom that made this whole check necessary."""
  bodies = [model.body(name).id]
  changed = True
  while changed:
    changed = False
    for b in range(model.nbody):
      if model.body_parentid[b] in bodies and b not in bodies:
        bodies.append(b)
        changed = True
  return [g for g in range(model.ngeom)
          if model.geom_bodyid[g] in bodies and model.geom_contype[g]]


def _bay_geoms(model, i):
  from pluggybot.rack.coupling import bay_prefix
  pre = bay_prefix(i)
  return [g for g in range(model.ngeom)
          if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
          .startswith(pre) and model.geom_contype[g]]


def _flank_clearance(model, data, bay):
  """How far a peg must rise from its rest to pass over the tray flanks."""
  from pluggybot.rack.coupling import HUB_PEG_Z, PEG_R, TRAY_VERTEX_DROP
  mujoco.mj_forward(model, data)
  flank_top = max(
    float(data.geom_xpos[g][2]) + _world_half_extents(model, data, g)[2]
    for g in bay
    if "tray" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)))
  return flank_top - (HUB_PEG_Z - TRAY_VERTEX_DROP + PEG_R)


def _stow_raise_band():
  """How high above its HUNG rest the peg actually rides during a stow.

  Two terms, and missing the first one is what made an early version of
  this test pass while home_draw.py still failed: `put_back` adds
  RETURN_CLEARANCE to the lift it is ALREADY carrying at, and a carried peg
  settles into the fork's V some 9-18 mm above where it hangs (SimNotes;
  measured at 11.2 mm on the home-world errand). So the real height is the
  sum, not RETURN_CLEARANCE alone -- 29-38 mm rather than 20.

  Read the pen's pass with that in mind: its ceiling is 39 mm, so the top
  of this band clears by only 1 mm -- but 38 is itself the pessimistic end
  of a range measured across ALL modules, and the pen's own raise is 31 mm
  with ~8 mm in hand. The 1 mm is slack between two conservative bounds,
  not the margin the robot actually flies with. If a heavier pen variant
  ever pushes the carry rise toward 18 mm, buy the headroom back by moving
  the rail further below the pen line (`PEN_RAIL_DZ`) -- that is free,
  because the pen line does not move with it.
  """
  from pluggybot.rack.swap import RETURN_CLEARANCE
  return RETURN_CLEARANCE + 0.009, RETURN_CLEARANCE + 0.018


def _fouls(model, data, name, mod, bay, dz):
  """Module<->bay contacts with the module raised dz above its hung rest.

  The rest height comes from the coupling's own constants rather than from
  data, so repeated calls cannot drift by reading back a height this
  function itself set.
  """
  from pluggybot.rack.coupling import (
    HUB_PEG_Z, PEG_ABOVE_BODY, PEG_R, TRAY_VERTEX_DROP,
  )
  adr = model.jnt_qposadr[model.body(name).jntadr[0]]
  z0 = HUB_PEG_Z - TRAY_VERTEX_DROP + PEG_R - PEG_ABOVE_BODY
  data.qpos[adr + 2] = z0 + dz
  mujoco.mj_forward(model, data)
  hits = set()
  for k in range(data.ncon):
    c = data.contact[k]
    pair = {c.geom1, c.geom2}
    if pair & mod and pair & bay:
      hits.add(tuple(sorted(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in pair)))
  return hits


@pytest.mark.parametrize("i,name", list(enumerate(
  ("module_lcd", "module_plug", "module_pen", "module_claw", "module_seed"))))
def test_return_clearance_fits_inside_the_bay_window(hub_model, i, name):
  """Issue #10. Setting a module down needs TWO clearances at once, and the
  pen had no height that satisfied both.

  To drop a peg into its tray V, the peg must first be carried over the tray
  FLANKS -- 14.7 mm above its resting height here. That is the floor.
  Whatever else the module carries must at the same time stay clear of the
  bay's bracket feet and columns, which hang just under the trays. That is
  the ceiling. `put_back` raises the module by RETURN_CLEARANCE and drives
  in, so RETURN_CLEARANCE must land strictly between the two.

  For four modules the ceiling is far away -- nothing of theirs reaches into
  the bracket band. The pen's rail and carriage did: they topped out 16 mm
  under the feet against a 14.7 mm floor, a window about 1 mm wide, and
  RETURN_CLEARANCE (20 mm) sat inside the foul band. The module rode in,
  jammed its rail on the bracket feet ~13 mm short of the tray line, and
  rode back out on the fork -- in room_hub and home_world alike.

  Swept geometrically rather than by running a stow, because that is what
  makes the number legible: this asserts the WINDOW, so it also fails for
  the sixth tool that hangs something new into the bracket band.
  """
  mod = set(_module_geoms(hub_model, name))
  bay = set(_bay_geoms(hub_model, i))
  assert mod and bay
  data = mujoco.MjData(hub_model)
  need = _flank_clearance(hub_model, data, bay)
  lo, hi = _stow_raise_band()
  assert lo > need, (
    f"the stow lifts the peg {lo * 1000:.1f} mm at least, which does not "
    f"clear the {name} bay's tray flanks ({need * 1000:.1f} mm)")
  for dz in (need + 0.001, lo, (lo + hi) / 2, hi):
    hits = _fouls(hub_model, data, name, mod, bay, dz)
    assert not hits, (
      f"{name} raised {dz * 1000:.1f} mm into its bay jams on it: "
      f"{sorted(hits)} -- the stow uses {lo * 1000:.0f}-{hi * 1000:.0f} mm, "
      f"so this height is one the robot actually passes through")


# Where a square figure leaves the pen carriage (measured in home_draw).
CARRIAGE_AFTER_A_FIGURE = 0.037


def test_a_parked_carriage_would_jam_the_pen_stow(hub_model):
  """The premise behind PenPlotter.carry_config centring the carriage.

  Pinned as a test rather than left as a comment, for the same reason the
  navigation tests pin the pure-pursuit orbit: if the bay ever grows enough
  room for an off-centre carriage, this test fails and tells us the centring
  is no longer load-bearing. Until then it documents WHY it is.

  Centred, the carriage block sits in the gap BETWEEN the bay's two tray
  brackets. Run out along the peg axis it swings into the y band where they
  hang, and the ceiling of the clear window drops below the height the stow
  actually uses.
  """
  mod = set(_module_geoms(hub_model, "module_pen"))
  bay = set(_bay_geoms(hub_model, 2))
  data = mujoco.MjData(hub_model)
  car = hub_model.joint("pen_carriage_joint").qposadr[0]
  _lo, hi = _stow_raise_band()
  data.qpos[car] = CARRIAGE_AFTER_A_FIGURE
  assert _fouls(hub_model, data, "module_pen", mod, bay, hi), (
    "an off-centre carriage no longer fouls the bay -- carry_config's "
    "centring may have stopped being necessary, or the bay changed")


def test_pen_stows_back_into_its_bay_after_its_carriage_has_moved(hub_model):
  """Issue #10 end to end, minus navigation and minus drawing.

  Both halves of the bug in one run, because both end the same way -- the
  module riding the fork back out instead of transferring to the trays:

    1. the rail, which fouled the bracket at every stow, drawn or not;
    2. the carriage left where a figure ended, which fouled it only after
       a drawing -- which is why the geometry fix alone still left
       home_draw.py reporting STOW: FAILED.

  Displacing the carriage by hand stands in for the drawing: it is the
  state a figure leaves behind, without paying for a figure.
  """
  from pluggybot.tools.drawing import PenPlotter
  from pluggybot.rack.swap import ARM_EXT
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  assert swap.module_state("module_pen")["on_fork"], "pen pick failed"

  car = hub_model.joint("pen_carriage_joint").qposadr[0]
  plotter = PenPlotter(hub_model, data, swap)
  plotter.ramp(plotter.pen_act, CARRIAGE_AFTER_A_FIGURE, settle=0.5)
  plotter.carry_config()
  assert abs(float(data.qpos[car])) < 0.003, (
    f"carry_config left the carriage at {float(data.qpos[car]) * 1000:+.1f} mm "
    f"-- centred is the stow pose")

  # carry_config stows the arm for driving; the swap deploys it again on
  # arrival, exactly as HubMission.swap_at_bay does.
  data.ctrl[hub_model.actuator("arm").id] = ARM_EXT
  swap._run(1.5, 0.0)

  swap.put_back()
  st = swap.module_state("module_pen")
  assert st["hung"], (
    f"the pen did not hang up: {st} -- rack-frame x/z off by "
    f"{(st['rack_frame'][0] - 0.09) * 1000:+.1f}/"
    f"{(st['rack_frame'][2] - 0.273) * 1000:+.1f} mm")
  assert not st["on_fork"], "the pen hung but the fork did not let go"


def test_charge_bay_press_connects(hub_model):
  """Nosing into the charge bay and pressing must land BOTH pogo pins on the
  bumper -- the rack-side charge criterion -- without shoving the rack."""
  import math
  from pluggybot.rack.coupling import CHARGE_BAY_Y, rack_charge_contact
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  yaw = math.pi
  data.qpos[0] = 0.60 + 0.08 * math.cos(yaw)
  data.qpos[1] = CHARGE_BAY_Y
  data.qpos[2] = 0.045
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(hub_model, data)
  swap.reckoner.x, swap.reckoner.y, swap.reckoner.theta = 0.60, CHARGE_BAY_Y, yaw
  swap._drive_until(0.32, 0.04, timeout=15.0)
  assert rack_charge_contact(hub_model, data), "pins not both on the bumper"
  rack = data.xpos[hub_model.body("rack").id]
  assert abs(float(rack[0])) < 0.005, "the press shoved the rack"
