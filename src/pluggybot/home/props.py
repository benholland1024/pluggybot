"""The house's loose props (issues #207, #215): the tower's three blocks in
the workshop's corner and the lab's bench with its two masses, emitted by
the house's generator (`home.world`), and the cube every other prop of the
house is sized by (`BLOCK_HALF`). Their jobs -- the tower and the bench --
went with the quadruped (#559, stage 5); the props stay in the house, whose
geometry the served world's continuation fingerprints, and each may come
back as a task the humanoids learn with their hands (#560).
"""

#: Near-rigid contact for what is gripped and stacked: a block carries it,
#: or what holds it is the solver's (CLAUDE.md, "A challenge is a task...").
GRIP_SOLIMP = "0.99 0.999 0.0001"

#: The blocks: a 26 mm cube, the size the rover's claw was measured against
#: (70 mm jaw span).
BLOCK_HALF = 0.013
BLOCK_MASS = 0.06
BLOCK_FRICTION = 1.2
BLOCKS = ("block_0", "block_1", "block_2")


def block_xml(name: str, x: float, y: float, tag_id: int | None = None,
              indent: str = "    ", mass: float = BLOCK_MASS) -> str:
  """One block as MJCF text, for the generator, with its AprilTag on every
  face where it has one (tags.BLOCK_TAG_IDS; the material is the world's
  `tagmat<id>`). `mass` is the bench's: the same cube at another weight."""
  solimp = " ".join(GRIP_SOLIMP.split() + ["0.9", "2.0"])
  skin = (f'material="tagmat{tag_id}"' if tag_id is not None
          else 'rgba="0.90 0.60 0.20 1"')
  return (f'{indent}<body name="{name}" pos="{x:.4f} {y:.4f} {BLOCK_HALF:.4f}">\n'
          f'{indent}  <freejoint/>\n'
          f'{indent}  <geom name="{name}_box" type="box" '
          f'size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" mass="{mass}" '
          f'friction="{BLOCK_FRICTION} 0.005 0.0001" solimp="{solimp}" '
          f'{skin}/>\n'
          f'{indent}</body>')


#: The bench: EXACTLY ONE BOX (hint `table`, the height is the TOP), a
#: 1.2 x 0.6 m worktop at 0.8 m against the lab's east wall, its long side
#: along the wall.
BENCH_HALF = (0.30, 0.60, 0.40)
BENCH_RGBA = "0.55 0.47 0.38 1"
#: Its two masses, the known and the one the bench's job hid: blocks on the
#: floor 1 m in front of its face (the bench faces -x, into the lab) and
#: half a metre apart, between the bench's tags (`home.areas`).
MASSES = ("mass_known", "mass_unknown")
MASS_KG = (0.100, 0.150)
MASS_OFFSETS = ((-1.0, -0.25), (-1.0, 0.25))


def bench_xml(bench_xy: tuple[float, float], prefix: str = "lab") -> str:
  """MJCF for the bench and its two masses, for the generator."""
  from pluggybot.perception.tags import MASS_TAG_IDS
  bx, by = bench_xy
  hx, hy, hz = BENCH_HALF
  out = (f'    <body name="{prefix}_bench" pos="{bx:.4f} {by:.4f} {hz:.4f}">\n'
         f'      <geom name="{prefix}_bench_geom" type="box" '
         f'size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="{BENCH_RGBA}"/>\n'
         f'    </body>')
  for name, (dx, dy), tag, mass in zip(MASSES, MASS_OFFSETS, MASS_TAG_IDS, MASS_KG):
    out += "\n" + block_xml(name, bx + dx, by + dy, tag, mass=mass)
  return out
