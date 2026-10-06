"""A scene compiled into a world of the robot's own (issue #466): its own
body -- the CAD a real robot has, `legs.model`'s -- on a floor with the tool
it carries on its fork, and what the document describes. Never the world's
(the fence: the package docstring).

Each part is a body of its own at its pose in the map, a child of the part
it rides; a joint is the part's, in its frame. What touches what is the
compiler's rule, not the document's: everything touches everything but a
moving part and the rigid group it is hinged to -- the parts joined to that
one without a joint, and every part fixed in the map is one group, the
map's -- where its hinge would otherwise jam. A catch is a force the
physics has no element for, set every step by `CompiledCatch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from pluggybot.imagination.scene import Catch, Joint, Scene, lever, rotation

#: THE IMAGINATION'S OWN CONTACT, chosen apart from the world's (MuJoCo's
#: defaults, which the house's furniture and the chest carry): imagined in
#: the world's engine WITH the world's contact model, a model transfers better
#: than it would to hardware (#465's caution), so the parts a document
#: describes touch each other on settings of their own -- stiffer than the
#: world's, at five physics steps' time constant, and wood on wood's
#: friction. ⚠ Against the robot, MuJoCo mixes a pair's settings (the larger
#: friction) and the robot's priority geoms -- the claw's pads, the fork, the
#: feet -- impose their own: the chest's probe touches the knob with the
#: pads alone, on the pads' settings in either world. ⚠ At two steps,
#: MuJoCo's floor, a lid resting on walls it touched jammed there and the
#: solver blew up (#466).
CONTACT = 'friction="0.5 0.005 0.0001" solref="0.01 1" solimp="0.95 0.99 0.001"'
#: ...and a joint's stop, likewise: the one setting the chest's probe meets.
LIMIT = 'solreflimit="0.01 1" solimplimit="0.95 0.99 0.001"'
#: A catch holds over its first millimetre at its point (the language's one
#: catch: a pull of `release` until the part is that far out).
CATCH_REACH_M = 0.001
#: The parts' names in the compiled world: the robot's own are bare.
PREFIX = "scene_"

#: The robot's world before a scene goes in: its physics step, and a floor
#: under it at the map's z = 0 -- the floor it lies on everywhere, on
#: MuJoCo's defaults as the house's is: on `CONTACT`, its belly and thighs
#: lay on a mix of the two, a solref of 0.015 (the review of #473).
WORLD_XML = """<mujoco model="imagination">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1"/>
  </worldbody>
</mujoco>"""


def _f(v: float) -> str:
  return repr(float(v))


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def _quat(rot: np.ndarray) -> np.ndarray:
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.ascontiguousarray(rot, dtype=float).ravel())
  return q


def robot_spec(carrying: str | None = None) -> mujoco.MjSpec:
  """The robot's world before a scene goes in: a floor, and on it the robot
  as its CAD has it with `carrying` on its fork (`legs.imagined.body_spec`,
  the body's half)."""
  from pluggybot.legs.imagined import body_spec
  spec = mujoco.MjSpec.from_string(WORLD_XML)
  spec.attach(body_spec(carrying), prefix="", frame=spec.worldbody.add_frame())
  return spec


@dataclass(frozen=True)
class CompiledCatch:
  """A catch in a compiled world: `release` N at its point's `lever` (m off
  a hinge's axis; 1 on a slide) toward the end of its joint's range where
  it was caught (`sign` +1 at the low end, -1 at the high), while the joint
  is within `CATCH_REACH_M` of it at its point; past that, nothing."""
  joint: str
  qadr: int
  dof: int
  sign: float
  lever: float
  release: float

  @property
  def reach(self) -> float:
    """How far its joint may go before it lets go, rad or m."""
    return CATCH_REACH_M / self.lever

  def force(self, q: float) -> float:
    """Its generalized force at joint coordinate `q`."""
    if self.sign * q > self.reach:
      return 0.0
    return -self.sign * self.release * self.lever

  def __call__(self, model, data) -> None:
    data.qfrc_applied[self.dof] = self.force(float(data.qpos[self.qadr]))


@dataclass
class Imagined:
  """A world to roll out: its model, what it sets every step (`hooks`, each
  `hook(model, data)` after a step, as a world's step hooks are), its scene
  joints' addresses by the document's ids, and the tool on the fork."""
  model: mujoco.MjModel
  hooks: tuple = ()
  joints: dict[str, tuple[int, int]] = field(default_factory=dict)
  carrying: str | None = None


def _groups(scene: Scene) -> tuple[dict[str, int], int | None]:
  """Each part's rigid group -- the parts joined to it without a joint --
  and the map's, the one every part fixed in the map is in (None where
  there is none). ⚠ The map joins what is fixed in it: as groups of their
  own, a box's walls written on nothing let the lid hinged on one of them
  sit on the others' top edges, where it jammed shut and was thrown (the
  review of #473)."""
  moved = {j.part for j in scene.joints}
  parent = {p.id: p.id for p in scene.parts}

  def root(x: str) -> str:
    while parent[x] != x:
      x = parent[x]
    return x
  fixed = [p.id for p in scene.parts if p.on is None and p.id not in moved]
  for pid in fixed[1:]:
    parent[root(pid)] = root(fixed[0])
  for p in scene.parts:
    if p.on is not None and p.id not in moved:
      parent[root(p.id)] = root(p.on)
  roots = sorted({root(p.id) for p in scene.parts})
  groups = {p.id: roots.index(root(p.id)) for p in scene.parts}
  return groups, (groups[fixed[0]] if fixed else None)


def scene_xml(scene: Scene) -> str:
  """The scene as a `<mujoco>` to attach at the map's origin."""
  joint_of = {j.part: j for j in scene.joints}
  rot = {p.id: rotation(p.euler) for p in scene.parts}
  pos = {p.id: np.asarray(p.pos) for p in scene.parts}

  def body(pid: str) -> str:
    p = scene.part(pid)
    if p.on is None:
      rel_p, rel_r = pos[pid], rot[pid]
    else:
      rp = rot[p.on]
      rel_p, rel_r = rp.T @ (pos[pid] - pos[p.on]), rp.T @ rot[pid]
    if p.shape == "cylinder":
      geom = f'type="cylinder" size="{_v(p.size[0] / 2, p.size[1] / 2)}"'
    else:
      geom = f'type="box" size="{_v(*(s / 2 for s in p.size))}"'
    out = [f'<body name="{PREFIX}{pid}" pos="{_v(*rel_p)}" quat="{_v(*_quat(rel_r))}">']
    j = joint_of.get(pid)
    if j is not None:
      at = rot[pid].T @ (np.asarray(j.at) - pos[pid])
      axis = rot[pid].T @ np.asarray(j.axis)
      rng = (f'limited="true" range="{_v(*j.range)}" {LIMIT}' if j.range is not None
             else 'limited="false"')
      out.append(f'<joint name="{PREFIX}{j.id}" type="{j.type}" pos="{_v(*at)}" '
                 f'axis="{_v(*axis)}" {rng} stiffness="{_f(j.stiffness)}" '
                 f'springref="{_f(j.slack)}" damping="{_f(j.damping)}" '
                 f'frictionloss="{_f(j.friction)}"/>')
    out.append(f'<geom name="{PREFIX}{pid}" {geom} mass="{_f(p.mass)}" {CONTACT}/>')
    out += [body(c.id) for c in scene.parts if c.on == pid]
    out.append("</body>")
    return "".join(out)

  roots = "".join(body(p.id) for p in scene.parts if p.on is None)
  groups, the_map = _groups(scene)
  excludes = []
  for j in scene.joints:
    on = scene.part(j.part).on
    hinged_to = groups[on] if on is not None else the_map
    if hinged_to is None:
      continue
    mine = [p.id for p in scene.parts if groups[p.id] == groups[j.part]]
    theirs = [p.id for p in scene.parts if groups[p.id] == hinged_to]
    excludes += [f'<exclude body1="{PREFIX}{a}" body2="{PREFIX}{b}"/>'
                 for a in mine for b in theirs]
  contact = f"<contact>{''.join(excludes)}</contact>" if excludes else ""
  return (f'<mujoco><compiler angle="radian" autolimits="true"/>'
          f'<worldbody>{roots}</worldbody>{contact}</mujoco>')


def _catch(model: mujoco.MjModel, scene: Scene, c: Catch) -> CompiledCatch:
  j: Joint = scene.joint(c.joint)
  jid = model.joint(f"{PREFIX}{j.id}").id
  return CompiledCatch(joint=j.id, qadr=int(model.jnt_qposadr[jid]),
                       dof=int(model.jnt_dofadr[jid]),
                       sign=1.0 if j.range[0] == 0.0 else -1.0, lever=lever(j, c.at),
                       release=c.release)


def compile_scene(scene: Scene, carrying: str | None = None) -> Imagined:
  """`scene` in a world of the robot's own (`robot_spec`)."""
  spec = robot_spec(carrying)
  spec.attach(mujoco.MjSpec.from_string(scene_xml(scene)), prefix="",
              frame=spec.worldbody.add_frame())
  model = spec.compile()
  joints = {}
  for j in scene.joints:
    jid = model.joint(f"{PREFIX}{j.id}").id
    joints[j.id] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
  return Imagined(model=model, hooks=tuple(_catch(model, scene, c) for c in scene.catches),
                  joints=joints, carrying=carrying)
