"""Slice B of issue #168: a validated `Tool` becomes a module -- on legs, the
arm's coupling's (#407).

Three emitters, in the generator's own vocabulary so a built tool inherits
every contact rule the hand-built ones have:

  face_xml      the parts as geoms in the module's frame, an actuator's
                load as a child body with the joint -- the module's `face`,
                beside the plate and the 220 mm peg (`legs.rack.tool_xml`)
  actuator_xml  one `<position>` per axis: forcerange is the part's own
                torque or thrust, kp is that force over a saturation
                error, so the servo a spec names is the servo that moves
  register      the tool's axes and sensors into `procedure/axes.py`'s
                registries, which is how a program `move()`s a verb the
                agent named -- no new language

...and the RIG: `rig()` puts the built module on a bay of the legs rack and
the arm's own fork (`legs.arm.fork_xml`) on a carrier, and runs the swap's
lines -- in under the peg, up, out, every axis to its far end and back,
over the bay, down, out -- answering the questions ToolPattern §5's build
sequence asks before a tool goes near the rack: does it hang (plumb, as
`rack.on_bay` asks), is it picked, does it conduct, does it work, does it
hang back -- each off the world, never off a command. A spec that
validates and fails the rig is a real finding (the validator is static;
the rig has gravity): the pen, built as the rover's, hung 16 deg off plumb
(#406). The carrier is rigid; the body's sway at a bay is the swap's own
measurement (`scripts/arm_spike.py --capture`).

  ⚠ ANGLE UNITS, MEASURED: a compiler with `angle="degree"` converts a
  joint's `range` and never an actuator's `ctrlrange`, and an attached
  spec's elements keep their own compiler's unit. Every emitter here
  compiles in RADIANS -- the rig, and the seam's child spec -- so a hinge's
  range is written in radians, as its ctrlrange is: written in degrees (the
  rover's worlds compiled in them), a 0-90 deg hinge had a 0-90 rad limit,
  none, and a heavy blade swung over the top (#407).
  `test_workshop_build.py` pins it.

...and `trial()`, the rig as the workshop's gate before a point moves.
"""

from __future__ import annotations

import math

import mujoco

from pluggybot.procedure import axes
from pluggybot.workshop.spec import FRAME, Placed, Tool

#: A position actuator's gain is the part's force over the error at which
#: a real one saturates: a hobby servo delivers full torque at a few
#: degrees, a screw at a few millimetres. kp = force / saturation.
HINGE_SATURATION_RAD = math.radians(6.0)
SLIDE_SATURATION_M = 0.005
#: Damping as a fraction of kp, the ratio the existing modules use
#: (claw 600/6, gate 800/12, pen 2000/80 -- 1-4 %).
KV_FRACTION = 0.02

RGBA_PRINTED = "0.85 0.85 0.80 1"
RGBA_PART = "0.30 0.32 0.36 1"


def _f(v: float) -> str:
  return f"{v:.4f}"


def _geom_xml(name: str, p: Placed, pos, euler_rad) -> str:
  rgba = RGBA_PRINTED if p.part.kind == "scaffold" else RGBA_PART
  euler = " ".join(_f(math.degrees(e)) for e in euler_rad)
  if p.shape == "cylinder":
    size = f"{_f(p.half[0])} {_f(p.half[2])}"
  else:
    size = " ".join(_f(h) for h in p.half)
  return (f'<geom name="{name}" type="{p.shape}" size="{size}" '
          f'pos="{" ".join(_f(x) for x in pos)}" euler="{euler}" '
          f'mass="{p.mass:.4f}" rgba="{rgba}"/>')


def joint_name(body: str, verb: str) -> str:
  return f"{body}_{verb}_joint"


def actuator_name(body: str, verb: str) -> str:
  return f"{body}_{verb}"


def face_xml(tool: Tool, body: str) -> str:
  """The tool's parts in the module's frame. A part on the frame is a geom
  of the module body; an actuator's load -- every part `on` it -- is a
  child body at the actuator's pose carrying the joint, so it moves as
  the joint moves and MuJoCo filters its contact with its parent.

  ⚠ AT ITS STOW (#407): each load is built turned (or slid) to its axis's
  stow and its joint's `ref` is the stow, so the module compiles hanging as
  the validator checked it, `qpos0` is the stow -- the rest a walk, a
  hang-back and a reset put it back to -- and the joint still reads in the
  spec's own units. Built at 0, a flap whose stow is -90 deg hung 14 deg
  off plumb and was lost from the moment it hung."""
  def emit(parent_id: str, indent: str) -> list[str]:
    out = []
    for p in tool.children(parent_id):
      out.append(indent + _geom_xml(f"{body}_{p.id}", p, p.pos, p.euler))
      if p.axis is not None:
        a = p.axis
        at = [float(x) for x in p.pos]
        quat = ""
        if a.kind == "slide":
          at = [x + d * a.stow for x, d in zip(at, a.direction)]
        else:
          s = math.sin(a.stow / 2)
          quat = (f' quat="{_f(math.cos(a.stow / 2))} '
                  + " ".join(_f(d * s) for d in a.direction) + '"')
        pos = " ".join(_f(x) for x in at)
        d = " ".join(_f(x) for x in a.direction)
        rng = f"{a.lo:.6f} {a.hi:.6f}"
        out.append(f'{indent}<body name="{body}_{p.id}_load" pos="{pos}"{quat}>')
        out.append(f'{indent}  <joint name="{joint_name(body, a.verb)}" '
                   f'type="{a.kind}" axis="{d}" range="{rng}" limited="true" '
                   f'ref="{a.stow:.6f}" damping="{a.force * KV_FRACTION:.4f}"/>')
        # a load with no parts still needs an inertia to be a body -- ⚠ and
        # one WITH parts must not get it: an explicit inertial replaces its
        # geoms' masses, and every moving part of a built tool was massless
        # in the sim (#407: a servo could never be shown too weak)
        if not tool.children(p.id):
          out.append(f'{indent}  <inertial pos="0 0 0" mass="0.001" '
                     f'diaginertia="1e-7 1e-7 1e-7"/>')
        out.extend(emit(p.id, indent + "  "))
        out.append(f"{indent}</body>")
    return out
  return "\n      ".join(emit(FRAME, ""))


def actuator_xml(tool: Tool, body: str) -> str:
  """One position servo per axis, at the part's own force."""
  out = []
  for p in tool.axes:
    a = p.axis
    assert a is not None
    sat = HINGE_SATURATION_RAD if a.kind == "hinge" else SLIDE_SATURATION_M
    kp = a.force / sat
    out.append(f'<position name="{actuator_name(body, a.verb)}" '
               f'joint="{joint_name(body, a.verb)}" kp="{kp:.3f}" '
               f'kv="{kp * KV_FRACTION:.3f}" '
               f'ctrlrange="{a.lo:.6f} {a.hi:.6f}" '
               f'forcerange="{-a.force:.3f} {a.force:.3f}"/>')
  return "\n    ".join(out)


def module_for(tool: Tool, peg: tuple[float, float, float], yaw: float = 0.0,
               body: str | None = None) -> tuple[str, str]:
  """The whole module hanging with its peg's axis at `peg` (world), facing
  the robot at work (`yaw`, the rack's), as the hand-built ones are
  (`legs.rack.tool_xml`): (its default class, its body). The face carries
  its parts' own masses; the plate and the peg are `rack.MODULE_MASS`."""
  from pluggybot.legs import rack as rk
  body = body or tool.body
  return rk.tool_default(body), rk.tool_xml(
    body, peg, yaw=yaw, mass=rk.MODULE_MASS, face=face_xml(tool, body),
    rgba="0.45 0.40 0.55 1")


def sensor_name(tool: Tool, placed: Placed) -> str:
  return f"{tool.name}.{placed.id}.contact"


def contact_sensors(tool: Tool) -> list[Placed]:
  """The parts a program may ask "am I touching something" of: a catalog
  sensor whose `sense` is `contact` (the bumper's microswitch, issue #199).
  A camera is a sensor too and has no such reading."""
  return [p for p in tool.parts
          if p.part.kind == "sensor" and p.part.capabilities.get("sense") == "contact"]


def register(tool: Tool) -> list[str]:
  """The tool's verbs and joints into the registries a program reads.
  Returns the names, so a retire (slice D) can take them out again.

  A contact sensor part becomes `<tool>.<id>.contact`: 1 while that part's
  geom touches anything outside the module -- the bumper's own criterion,
  through `axes.geom_contact` -- so a tool that carries a microswitch can
  feel for a wall the way the chassis does. No new sensing: the contact
  list already says it."""
  names = []
  for p in tool.axes:
    a = p.axis
    assert a is not None
    unit = "rad" if a.kind == "hinge" else "m"
    name = f"{tool.name}.{a.verb}"
    axes.register_axis(axes.Axis(
      name, a.lo, a.hi, a.speed, unit,
      f"{tool.name}'s {a.verb}: {p.part.name} ({a.kind})",
      requires=tool.body, actuator=actuator_name(tool.body, a.verb),
      run=axes.ramped(actuator_name(tool.body, a.verb))))
    jn = joint_name(tool.body, a.verb)
    axes.register_sensor(axes.Sensor(
      name, lambda life, jn=jn: float(
        life.data.qpos[life.model.joint(jn).qposadr[0]]),
      f"{tool.name}'s {a.verb}, measured ({unit})", requires=tool.body))
    names.append(name)
  for p in contact_sensors(tool):
    name = sensor_name(tool, p)
    geom = f"{tool.body}_{p.id}"
    axes.register_sensor(axes.Sensor(
      name, lambda life, geom=geom: axes.geom_contact(life, geom),
      f"{tool.name}'s {p.id}: 1 while the {p.part.name} touches something "
      "outside the module", requires=tool.body))
    names.append(name)
  return names


def unregister(body: str) -> list[str]:
  """Take a retired BUILT tool's verbs out of the registries. A hand-built
  module's axes (the pen's carriage, the claw's jaws) stay: they are the
  language's, not the workshop's, and a retired pen leaves them gated on a
  module that is no longer on the rack."""
  from pluggybot.rack.tags import MODULE_TAG_IDS
  if body in MODULE_TAG_IDS:
    return []
  gone = [n for n, a in axes.AXES.items() if a.requires == body]
  for n in gone:
    del axes.AXES[n]
    axes.SENSORS.pop(n, None)
  # ...and the sensors that are not an axis's twin: a contact sense.
  felt = [n for n, s in axes.SENSORS.items() if s.requires == body and n not in gone]
  for n in felt:
    del axes.SENSORS[n]
  return gone + felt


# ---- the rig ----------------------------------------------------------------

#: The rig's carrier moves the fork along the swap's lines at the arm's
#: `FORK_V`, and holds each stop this long, s.
RIG_HOLD_S = 0.4
RIG_SETTLE_S = 1.5


def rig_xml(tool: Tool, body: str = "tool", dy: float = 0.0) -> str:
  """The rig's world: a floor, the legs rack with the tool on its middle
  bay, and the arm's fork on a carrier welded to a mocap body, standing
  off the bay as the arm's would at the working pose -- `dy` across it, as
  a walk-in stops off the bay's middle."""
  from pluggybot.legs import arm as am
  from pluggybot.legs import rack as rk
  spec = am.ArmSpec()
  default, module = module_for(tool, rk.bay_peg(rk.DEFAULT, 1), body=body)
  f = spec.fork
  # the plate turned to face the rack (-x), its V vertex at the carrier's
  # origin; the carrier starts at the standoff, `FORK_DROP` under the peg
  x0, z0 = am.STANDOFF, rk.DEFAULT.peg_z - am.FORK_DROP
  return f"""<mujoco model="rig"><compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>{default}{am.ARM_DEFAULTS.format(friction=0.01, tube=am.TUBE_R, fork_kg=am.FORK_GEOM_KG)}</default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    {rk.rack_xml(rk.DEFAULT, name="rig_rack", tags=False)}
    {module}
    <body name="carrier_mocap" mocap="true" pos="{x0:.4f} {dy:.4f} {z0:.4f}"/>
    <body name="carrier" pos="{x0:.4f} {dy:.4f} {z0:.4f}" childclass="arm">
      <freejoint/>
      <body name="rig_plate" pos="0 0 0" quat="0 0 0 1">
        <geom name="rig_plate_box" type="box" size="0.03 0.03 0.012" pos="{-f.vertex_x + 0.02:.4f} 0 {-f.vertex_z - 0.012:.4f}" mass="0.08"/>
        <body pos="{-f.vertex_x:.4f} 0 {-f.vertex_z:.4f}">{am.fork_xml(spec, prefix="rig_")}</body>
      </body>
    </body>
  </worldbody>
  <equality><weld body1="carrier_mocap" body2="carrier" solref="0.004 1"/></equality>
  <actuator>{actuator_xml(tool, body)}</actuator>
</mujoco>"""


def rig(tool: Tool, dy: float = 0.0) -> tuple[dict, list]:
  """Hang, pick, conduct, work, hang back: the answers the build sequence
  asks (the module docstring), each off the world, the fork `dy` across the
  bay. Returns the record and no frames."""
  from pluggybot.legs import arm as am
  from pluggybot.legs import rack as rk
  model = mujoco.MjModel.from_xml_string(rig_xml(tool, dy=dy))
  data = mujoco.MjData(model)
  tool_bid = model.body("tool").id
  dt = model.opt.timestep
  for p in tool.axes:
    data.ctrl[model.actuator(actuator_name("tool", p.axis.verb)).id] = p.axis.stow

  def step(n: int) -> None:
    for _ in range(n):
      mujoco.mj_step(model, data)

  def line(x: float, z: float) -> None:
    x0, _, z0 = (float(v) for v in data.mocap_pos[0])
    n = max(1, int(math.hypot(x - x0, z - z0) / am.FORK_V / dt))
    for k in range(1, n + 1):
      data.mocap_pos[0] = (x0 + (x - x0) * k / n, dy, z0 + (z - z0) * k / n)
      step(1)
    step(int(RIG_HOLD_S / dt))

  def ramp(act: int, target: float, speed: float) -> None:
    cur = float(data.ctrl[act])
    n = max(int(abs(target - cur) / speed / dt), 1)
    for k in range(1, n + 1):
      data.ctrl[act] = cur + (target - cur) * k / n
      step(1)
    step(int(0.5 / dt))

  step(int(RIG_SETTLE_S / dt))
  z = data.xmat[tool_bid].reshape(3, 3)[2, 2]
  tilt = math.degrees(math.acos(min(1.0, float(z))))
  hangs = rk.on_bay(model, data, "tool", rk.DEFAULT, 1)
  peg_z = rk.DEFAULT.peg_z
  line(0.0, peg_z - am.FORK_DROP)                    # in under the peg
  line(0.0, peg_z - am.FORK_DROP + am.LIFT)          # up off the trays
  line(am.BACK_OUT, peg_z - am.FORK_DROP + am.LIFT)  # out
  picked = not rk.on_bay(model, data, "tool", rk.DEFAULT, 1)
  conducts = rk.tool_power(model, data, "tool", prefix="rig_")
  works = {}
  for p in tool.axes:
    a = p.axis
    act = model.actuator(actuator_name("tool", a.verb)).id
    q = model.joint(joint_name("tool", a.verb)).qposadr[0]
    far = a.hi if abs(a.hi - a.stow) >= abs(a.lo - a.stow) else a.lo
    ramp(act, far, a.speed)
    reached = abs(float(data.qpos[q]) - far) < (0.05 if a.kind == "hinge" else 0.002)
    ramp(act, a.stow, a.speed)
    back = abs(float(data.qpos[q]) - a.stow) < (0.05 if a.kind == "hinge" else 0.002)
    works[a.verb] = reached and back
  held = rk.tool_power(model, data, "tool", prefix="rig_")
  line(0.0, peg_z - am.FORK_DROP + am.LIFT)          # over the bay
  line(0.0, peg_z - am.FORK_DROP)                    # down into the trays
  line(am.BACK_OUT, peg_z - am.FORK_DROP)            # out, empty
  step(int(RIG_SETTLE_S / dt))
  stowed = rk.on_bay(model, data, "tool", rk.DEFAULT, 1)
  return {
    "hangs": hangs, "tiltDeg": round(tilt, 2), "picked": picked,
    "conducts": bool(conducts["powered"]), "poles": conducts, "works": works,
    "held_through_use": bool(held["powered"]), "stowed": stowed,
    "ok": hangs and picked and conducts["powered"] and all(works.values())
          and held["powered"] and stowed,
  }, []


def trial(tool: Tool, records: list | None = None) -> list[str]:
  """The rig as the workshop's last gate, BEFORE a point moves (#407): the
  module tried on a bay with the fork on its middle, then at the walk-in's
  line-up gate either side (`rack.LINEUP_ACROSS`). The first failure's
  reason, or `[]`; each rig's record appended to `records` when given. The
  validator is static, and tools it admitted jammed on the rack's rail or
  tipped off the trays; hung, a tool that cannot hang is lost from the
  moment it is hung."""
  from pluggybot.legs import rack as rk
  for dy in (0.0, rk.LINEUP_ACROSS, -rk.LINEUP_ACROSS):
    rec, _ = rig(tool, dy=dy)
    if records is not None:
      records.append({"dy": dy, **rec})
    if rec["ok"]:
      continue
    stuck = [verb for verb, ok in rec["works"].items() if not ok]
    why = (f"hung on the trays it does not seat (it tilts {rec['tiltDeg']:g} deg)"
           if not rec["hangs"] else
           "the fork did not lift it off the trays" if not rec["picked"] else
           "seated on the fork it did not conduct" if not rec["conducts"] else
           f"its {', '.join(stuck)} did not reach its far end and come back" if stuck else
           "it lost the fork's contacts while its axes moved" if not rec["held_through_use"]
           else "hung back, it did not seat on the trays")
    where = ("with the fork on the bay's middle" if dy == 0.0
             else f"with the fork {dy * 1000:+.0f} mm across the bay, where a walk-in may stop")
    return [f"rig: {why} {where} -- tried on a bench rack before anything was bought"]
  return []


__all__ = ["face_xml", "actuator_xml", "module_for", "register", "rig", "trial"]
