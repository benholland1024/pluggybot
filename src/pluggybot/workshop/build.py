"""Slice B of issue #168: a validated `Tool` becomes a module.

Three emitters, all in the generator's own vocabulary so a built tool
inherits every contact rule `rack/coupling.py` already applies:

  face_xml      the parts as geoms in the module's frame, an actuator's
                load as a child body with the joint -- injected into
                `module_xml` as its `face`, beside the plate and the peg
  actuator_xml  one `<position>` per axis: forcerange is the part's own
                torque or thrust, kp is that force over a saturation
                error, so the servo a spec names is the servo that moves
  register      the tool's axes and sensors into `procedure/axes.py`'s
                registries, which is how a program `move()`s a verb the
                agent named -- no new language

...and the RIG: `rig()` runs the built module through the coupling spike
(`scene_xml(module=)`) and answers the four questions ToolPattern §5's
build sequence asks before a tool goes near the rack -- does it hang, is it
picked, does it conduct, does it stow -- each measured off the world, never
off a command. A spec that validates and fails the rig is a real finding
(the validator is static; the rig has gravity), and the rig is what the
fabrication cost of slice D is paid against.

  ⚠ ANGLE UNITS, MEASURED: with the worlds' default `angle="degree"` the
  compiler converts a joint's `range` to radians and does NOT convert an
  actuator's `ctrlrange` -- a hinge position actuator's ctrlrange is
  radians whatever the compiler says. So a hinge's joint range is written
  in degrees and its ctrlrange in radians. `test_workshop_build.py` pins it.

  ⚠ THE RIG'S YAW IS NOT THE SPIKE'S. Measured on its first flight: at the
  standoff the fork's V-plates sit BESIDE the tray's V-plates with 4 mm of
  y clearance (FORK_Y - 6 mm against TRAY_Y + 8 mm), and 2° of yaw closes
  it -- the two interlock and the peg, seated and conducting, cannot be
  carried out. The spike's wall approach never has the two Vs at one x, so
  its 2° pick tolerance does not transfer. That is a fact about the
  coupling's pick geometry, not about any tool: the rig runs at the yaw
  the robot delivers (`bay_fix`, ~0.4°) and yaw tolerance stays the spike's
  and `scripts/swap_spike.py`'s measurement.

  ⚠ NOT HERE: the identity tag. It is visual-only and the spike has no
  camera; a built module hung on a REAL rack (slice C) gets its tag and
  its bay there, where the PNG is written once.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pluggybot.procedure import axes
from pluggybot.rack import coupling
from pluggybot.rack.coupling import (
  HUB_PEG_Z, LIFT_STEP, PUSH_FORCE, RACK_HANG_X, START_X, module_xml, scene_xml,
)
from pluggybot.workshop.spec import FRAME, Placed, Tool

#: A position actuator's gain is the part's force over the error at which
#: a real one saturates: a hobby servo delivers full torque at a few
#: degrees, a screw at a few millimetres. kp = force / saturation.
HINGE_SATURATION_RAD = math.radians(6.0)
SLIDE_SATURATION_M = 0.005
#: Damping as a fraction of kp, the ratio the existing modules use
#: (claw 600/6, gate 800/12, pen 2000/80 -- 1-4 %).
KV_FRACTION = 0.02

#: The spike fork's pole plates, by the names `scene_xml` gives them
#: (`_v_notch_xml("fork_l_", ...)`); the robot's are `FORK_POLE_GEOMS`.
SPIKE_POLES = {"l": ("fork_l_a", "fork_l_b"), "r": ("fork_r_a", "fork_r_b")}

#: How far the carrier advances to put the fork's V vertex (carrier-local
#: x = -0.055) under the peg (x = 0): what the robot's measured standoff
#: does, rather than pushing until a wall stops it. Measured on the rig's
#: first flight: the spike's stop is 28 mm past the peg and lifts it on
#: the prong BAR, not in the V, so the poles never touch -- and a stand-in
#: wall 4 mm behind the plate leaves no room for a face on the wall side.
#: The push itself stays the spike's 10 N: at the standoff the fork never
#: reaches the plate, so the force only sets the approach speed, and at a
#: yaw error it is what squares the fork against the trays by contact (at
#: 0.8 N the carrier stalled 17 mm short at 2° and never picked).
RIG_ADVANCE_M = START_X - 0.055
RIG_PUSH_N = PUSH_FORCE

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
  the joint moves and MuJoCo filters its contact with its parent."""
  def emit(parent_id: str, indent: str) -> list[str]:
    out = []
    for p in tool.children(parent_id):
      out.append(indent + _geom_xml(f"{body}_{p.id}", p, p.pos, p.euler))
      if p.axis is not None:
        a = p.axis
        pos = " ".join(_f(x) for x in p.pos)
        d = " ".join(_f(x) for x in a.direction)
        if a.kind == "hinge":
          rng = f"{math.degrees(a.lo):.3f} {math.degrees(a.hi):.3f}"
        else:
          rng = f"{_f(a.lo)} {_f(a.hi)}"
        out.append(f'{indent}<body name="{body}_{p.id}_load" pos="{pos}">')
        out.append(f'{indent}  <joint name="{joint_name(body, a.verb)}" '
                   f'type="{a.kind}" axis="{d}" range="{rng}" limited="true" '
                   f'damping="{a.force * KV_FRACTION:.4f}"/>')
        # a load with no parts still needs an inertia to be a body
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


def module_for(tool: Tool, x: float, y: float, peg_z: float,
               yaw_deg: float = 0.0, body: str | None = None,
               tag: str = "") -> str:
  """The whole module at a station, through `module_xml` like the five
  built by hand: plate, peg and the tool's face."""
  body = body or tool.body
  return module_xml(body, x, y, peg_z, "0.45 0.40 0.55 1", face=face_xml(tool, body) + tag,
                    yaw_deg=yaw_deg)


def register(tool: Tool) -> list[str]:
  """The tool's verbs and joints into the registries a program reads.
  Returns the names, so a retire (slice D) can take them out again."""
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
  return names


# ---- the rig ----------------------------------------------------------------

def rig(tool: Tool, dy: float = 0.0, dz: float = 0.0, yaw_deg: float = 0.0,
        n_frames: int = 0, camera: str = "side") -> tuple[dict, list[np.ndarray]]:
  """Hang, pick, conduct, stow: the four answers the build sequence asks.

  The module hangs in the spike's trays; the carrier slides in, lifts,
  carries it out (PICKED: the body rose and left with the fork), the poles
  are read off the contact list (CONDUCTS: both, via `module_power_state`
  against the spike's own fork plates), every axis is run to its far end
  and back to its stow (WORKS: the joint got there, measured), and the
  carrier hangs the module back up and leaves (STOWED: peg near its rest
  height, body near the centreline, fork clear). HANGS is the settle
  before any of it: the module did not fall out of the trays on its own.
  """
  model = mujoco.MjModel.from_xml_string(
    scene_xml(dy, dz, yaw_deg, face=face_xml(tool, "tool"),
              actuators=actuator_xml(tool, "tool"), push=RIG_PUSH_N,
              back_x=-RACK_HANG_X - 0.004, peg_z=HUB_PEG_Z))
  data = mujoco.MjData(model)
  push = model.actuator("push").id
  lift = model.actuator("lift").id
  tool_bid = model.body("tool").id
  renderer = mujoco.Renderer(model, 360, 480) if n_frames else None
  frames: list[np.ndarray] = []
  # the filmstrip's clock: the fixed phases plus each axis out and back
  total_time = 26.0 + sum(
    2 * (abs(p.axis.hi - p.axis.stow) + abs(p.axis.lo - p.axis.stow)) / p.axis.speed
    + 2 * 0.5 for p in tool.axes)

  adv = model.joint("advance").qposadr[0]

  def step(n, ctrl_push, ctrl_lift, until: float | None = None):
    """`until`: stop pushing once the carrier has advanced this far -- the
    measured standoff, not a wall."""
    for _ in range(n):
      if until is not None and float(data.qpos[adv]) >= until:
        ctrl_push = 0.0
      data.ctrl[push] = ctrl_push
      data.ctrl[lift] = ctrl_lift
      mujoco.mj_step(model, data)
      if renderer is not None and len(frames) < n_frames and \
         data.time > len(frames) * total_time / n_frames:
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

  def ramp(act: int, target: float, speed: float, settle_s: float = 0.5):
    cur = float(data.ctrl[act])
    steps = max(int(abs(target - cur) / speed / model.opt.timestep), 1)
    for k in range(steps):
      data.ctrl[act] = cur + (target - cur) * (k + 1) / steps
      step(1, data.ctrl[push], data.ctrl[lift])
    step(int(settle_s / model.opt.timestep), data.ctrl[push], data.ctrl[lift])

  # stow every axis before anything (the parked pose is the stow pose)
  for p in tool.axes:
    data.ctrl[model.actuator(actuator_name("tool", p.axis.verb)).id] = p.axis.stow
  step(1500, 0.0, 0.0)
  z_rest = float(data.xpos[tool_bid][2])
  hangs = abs(z_rest - (HUB_PEG_Z - coupling.TRAY_VERTEX_DROP + coupling.PEG_R
                        - coupling.PEG_ABOVE_BODY)) < 0.006

  step(5000, RIG_PUSH_N, 0.0, until=RIG_ADVANCE_M)   # slide under, to the standoff
  step(2000, 0.0, LIFT_STEP)                  # lift off the trays
  step(6500, -RIG_PUSH_N * 0.4, LIFT_STEP)    # carry out
  picked = (float(data.xpos[tool_bid][2]) - z_rest > 0.005
            and float(data.xpos[tool_bid][0]) > 0.05)
  conducts = coupling.module_power_state(model, data, "tool", poles=SPIKE_POLES)

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
  still_held = coupling.module_power_state(model, data, "tool", poles=SPIKE_POLES)

  step(5000, RIG_PUSH_N, LIFT_STEP, until=RIG_ADVANCE_M)   # return, high, to the standoff
  step(2000, 0.0, 0.0)                        # lower into the trays
  step(4000, -RIG_PUSH_N * 0.4, 0.0)          # leave empty
  tp = data.xpos[tool_bid]
  stowed = (abs(float(tp[2]) - z_rest) < 0.006 and abs(float(tp[0])) < 0.012
            and abs(float(tp[1])) < 0.02)
  if renderer is not None:
    renderer.close()
  return {
    "hangs": hangs,
    "picked": picked,
    "conducts": bool(conducts["powered"]),
    "poles": conducts,
    "works": works,
    "held_through_use": bool(still_held["powered"]),
    "stowed": stowed,
    "ok": hangs and picked and conducts["powered"] and all(works.values())
          and still_held["powered"] and stowed,
  }, frames


__all__ = ["face_xml", "actuator_xml", "module_for", "register", "rig",
           "SPIKE_POLES"]
