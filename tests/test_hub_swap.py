"""Guards for the fork robot + hub world integration (milestone 8)."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.hub.coupling import HUB_STATION_YS, TOOL_HALF_X
from pluggybot.hub.swap import HubSwap

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
  from pluggybot.hub.coupling import module_power_contact, module_power_state
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
  from pluggybot.hub.coupling import module_power_state
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap.pick()
  assert swap.module_state("module_lcd")["on_fork"], "pick failed"

  def drive(seconds, v, w):
    dropouts = {"left": 0, "right": 0, "steps": 0}
    tl, tr = wheel_targets(v, w)
    for _ in range(round(seconds / hub_model.opt.timestep)):
      swap._step_once(tl, tr)
      st = module_power_state(hub_model, data)
      dropouts["steps"] += 1
      for pole in ("left", "right"):
        if not st[pole]:
          dropouts[pole] += 1
    return dropouts

  # Turn away from the rack first, then a net-zero haul: out, back, turn,
  # counter-turn. (Driving straight forward after a pick rams the hub.)
  drive(2.7, 0.0, 1.2)
  total = {"left": 0, "right": 0, "steps": 0}
  for seconds, v, w in ((2.0, 0.25, 0.0), (1.0, 0.0, 0.0), (2.0, -0.25, 0.0),
                        (2.0, 0.0, 1.2), (2.0, 0.0, -1.2)):
    d = drive(seconds, v, w)
    for k in total:
      total[k] += d[k]

  assert swap.module_state("module_lcd")["on_fork"], "dropped the tool"
  for pole in ("left", "right"):
    frac = total[pole] / total["steps"]
    assert frac < 0.01, (
      f"{pole} pole lost contact for {frac:.1%} of the haul "
      f"({total[pole]}/{total['steps']} steps) -- the tool browns out "
      f"while the robot drives")


def test_bay_tag_ids_pair_by_index(hub_model):
  """Bay <-> tag pairing must survive a third bay. This was a hardcoded
  two-bay equality check, which would have silently steered every bay-C swap
  onto bay B's marker -- the exact class of bug real AprilTags were adopted
  to make impossible."""
  from pluggybot.hub.coupling import HUB_STATION_YS, bay_tag_id
  from pluggybot.hub.tags import BAY_TAG_IDS
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
  from pluggybot.hub.coupling import (
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
  from pluggybot.hub.coupling import HUB_STATION_YS, PEN_TRAVEL
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


def test_charge_bay_press_connects(hub_model):
  """Nosing into the charge bay and pressing must land BOTH pogo pins on the
  bumper -- the rack-side charge criterion -- without shoving the rack."""
  import math
  from pluggybot.hub.coupling import CHARGE_BAY_Y, rack_charge_contact
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
