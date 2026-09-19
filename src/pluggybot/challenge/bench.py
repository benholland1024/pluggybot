"""The physics bench (issue #215): the props for #227, as geometry.

"Find the unknown mass and record it" (#227) is a challenge in
Challenges.md's sense -- criteria written before the robot sees it, open in
method -- and this module is the half of it a world can hold before the
grader exists: a workbench against the lab's east wall and two cubes on
the floor in front of it, a KNOWN mass and an UNKNOWN one. The cubes are
the tower's blocks (challenge/stack.py: 26 mm, `GRIP_SOLIMP`, a tag on
every face -- ids 23 and 24, `tags.MASS_TAG_IDS`) because that cube is the
one grasp the claw has been proven on; only the mass differs.

⚠ `UNKNOWN_MASS_KG` IS A PLACEHOLDER, NOT THE ANSWER. #227 draws the
unknown from a bank rotated per offer and writes it into
`model.body_mass` (then `mj_setConst`) at claim time, and keeps it out of
every context the robot sees. The number here is only what the cube weighs
between offers, and nothing may read it as a truth. The known mass is the
job's own statement and is told to the robot in the offer.
"""

from pluggybot.challenge.stack import block_xml
from pluggybot.rack.tags import MASS_TAG_IDS

#: The bench: EXACTLY ONE BOX (hint `table`, the height is the TOP), a
#: 1.2 x 0.6 m worktop at 0.8 m. Its long side runs along the wall.
BENCH_HALF = (0.30, 0.60, 0.40)
BENCH_RGBA = "0.55 0.47 0.38 1"

KNOWN_MASS_KG = 0.100
UNKNOWN_MASS_KG = 0.150

MASSES = ("mass_known", "mass_unknown")
#: Where the two cubes START, from the bench's centre: on the floor a metre
#: in front of it (the bench faces -x, into the lab) and half a metre apart,
#: on no route the robot needs. The job is graded wherever they end up.
MASS_OFFSETS = ((-1.0, -0.5), (-1.0, 0.5))


def bench_xml(bench_xy: tuple[float, float], prefix: str = "lab") -> str:
  """MJCF for the bench and its two masses, for a generator."""
  bx, by = bench_xy
  hx, hy, hz = BENCH_HALF
  out = (f'    <body name="{prefix}_bench" pos="{bx:.4f} {by:.4f} {hz:.4f}">\n'
         f'      <geom name="{prefix}_bench_geom" type="box" '
         f'size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="{BENCH_RGBA}"/>\n'
         f'    </body>')
  for name, (dx, dy), tag, mass in zip(MASSES, MASS_OFFSETS, MASS_TAG_IDS,
                                      (KNOWN_MASS_KG, UNKNOWN_MASS_KG)):
    out += "\n" + block_xml(name, bx + dx, by + dy, tag, mass=mass)
  return out


def bench_center(model, prefix: str = "lab") -> tuple[float, float]:
  bid = model.body(f"{prefix}_bench").id
  return (float(model.body_pos[bid][0]), float(model.body_pos[bid][1]))
