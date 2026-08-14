"""Protocol version stamp + the dynamic-body census both emitters share.

The scene JSON ships every body once; telemetry frames re-ship only the
bodies that can move. Both sides must agree on which bodies those are and
which belong to the robot, so the census lives here, in one place.

Bump PROTOCOL_VERSION whenever either artifact's shape changes. The website
repo vendors fixture copies stamped with this version, so a bump is a
deliberate two-repo event -- never a side effect of an unrelated edit.
"""

PROTOCOL_VERSION = "0.1.0"

# The robot's root body. The planned multi-robot refactor (mjSpec attach with
# a namespace prefix per robot) will generalize this to a prefix; until then
# there is exactly one robot and it is called this everywhere.
ROBOT_ROOT = "pluggybot"


def dynamic_flags(model) -> list[bool]:
  """Per-body: can this body move in the world frame?

  A body is dynamic iff any joint sits between it and the world -- its own
  or an ancestor's. Only these need per-frame poses; walls, outlets and the
  rack frame ship once in the scene description and never again.
  """
  dyn = [False] * model.nbody
  for b in range(1, model.nbody):
    # bool(): body_jntnum is a numpy array, and `or` hands back numpy.bool_,
    # which json.dumps refuses -- caught the first time the scene was written
    dyn[b] = bool(dyn[model.body_parentid[b]] or model.body_jntnum[b] > 0)
  return dyn


def robot_body_ids(model) -> set[int]:
  """Ids of every body in the robot's subtree (ROBOT_ROOT and below)."""
  root = model.body(ROBOT_ROOT).id
  ids: set[int] = set()
  for b in range(model.nbody):
    x = b
    while x not in (0, root):
      x = int(model.body_parentid[x])
    if x == root:
      ids.add(b)
  return ids


def body_census(model) -> tuple[list[str], list[str]]:
  """(robot, world) name lists of the DYNAMIC bodies, in model order.

  The split mirrors the frame shape: the robot's own bodies stream under
  "robots"/<name>; shared dynamic bodies (the rack, the modules) stream
  under "world", because they are nobody's limbs -- either robot may move
  them once the shared world lands.
  """
  dyn = dynamic_flags(model)
  rob = robot_body_ids(model)
  robot = [model.body(b).name for b in range(model.nbody)
           if dyn[b] and b in rob]
  world = [model.body(b).name for b in range(model.nbody)
           if dyn[b] and b not in rob]
  return robot, world
