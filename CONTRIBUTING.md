# Contributing to PluggyBot

Start here if you want to work on PluggyBot, or to help it in another way.
The [README](README.md) says what the project is and how to run one scene.

## Who writes the code

Most of the code and the docs in this repository are written by Claude Code
sessions. Ben Holland decides what is built, and each decision is written
in an issue before the code is. Issue #465 holds the current order of work
and the decisions behind it.

[CLAUDE.md](CLAUDE.md) is the file that every session reads before it
starts. It is long because it holds every rule that an agent must not
break. Each rule comes with the number behind it and the document where its
story is told. Every session loads the whole file, so it keeps the rules
short and leaves the stories to `docs/`. You do not need to read it first.
The docs that the README lists are a better start.

## How the repository is laid out

The code in `src/pluggybot/` is divided by domain, not by when it was
built. A module goes where its concern lives. The rule itself is in
CLAUDE.md's "Conventions".

| Folder | What it holds |
|---|---|
| `humanoid/` | It holds the body: the model, each part with its datasheet, the code that runs the trained policies, and the code that reaches, walks and charges. |
| `perception/` | It holds the sensors, such as the LIDAR, the depth camera and the IMU, and the AprilTags the robot reads. |
| `mapping/` | It holds the map the robot lays from its LIDAR, the scan matching that keeps the map true, the searches for a route, and the places the robot finds. |
| `behavior/` | It plans a walk over the map, and holds the drive law that steers the body along it. |
| `home/` | It holds the world: the generator of the neighborhood, its boards, its props and its counters. |
| `activity/` | It holds the mechanisms that own world state, such as the charging stations, the chest, the drawers, the door and the dishware. |
| `mind/` | It holds the language model's side: what the model is shown, the menu it answers from, its memory and its own rules. |
| `procedure/` | It holds the small language the robots write procedures in, and its verbs. |
| `economy/` | It holds the jobs, their scoring, the points ledger and the energy prices, with their data files. |
| `mission/` | It holds the errands: the work queued for a robot to run. |
| `challenge/` | It holds one module for each graded job, and the reference solutions that pass them. |
| `evaluation/` | It holds the measurement: the arms, the qualities, and the kit for graded attempts. |
| `session/` | It flies a person's procedure in a world of its own. |
| `telemetry/` | It streams the world to the website, and writes its scenes and recordings. |

A few modules sit at the top of `src/pluggybot/`. `lifecycle.py` is the
robot's day loop, which ties the domains together. `body.py` is the one way
the loop reaches a body. `navigator.py` walks a body along its plans and
keeps it clear of the other robot. `tick.py` steps the physics, and
`pair.py` runs two robots in one loop.

The rest of the repository:

| Folder | What it holds |
|---|---|
| `scripts/` | It holds the demos and the probes. Every script takes `--help`. |
| `tests/` | It holds the tests, in one flat folder. |
| `docs/` | It holds the design and the record. The README's table says which doc to read first. |
| `models/` | It holds the generated MuJoCo models, the trained policies and the tag images. |
| `protocol/` | It holds the format of the stream, and the fixtures that the website copies. |
| `training/` | It trains the policies. It is a uv project of its own. |
| `deploy/`, `Dockerfile` | They build the image that serves the live world. |

## How a change is made

A change starts as an issue. The issue says what is wrong or missing, and
the decision is written there before the code is. Before you change a part
of the code, read the doc that owns it. The table at the top of CLAUDE.md
says which doc that is.

These rules hold for every change:

- **A fixed bug becomes a test.** The test must fail without the fix, and
  it is written as cheaply as it can be: a direct call or a stub, not a
  whole flight, unless the claim is about physics.
  [docs/Testing.md](docs/Testing.md) shows how.
- **The test suite has a budget.** A change that would take the whole
  suite past ten minutes on a quiet machine, or fifteen on a busy one,
  needs Ben's approval. So does any test that leaves the default run, such
  as a skip, a gate or a new marker. CLAUDE.md's "Working style" keeps the
  rule, and the suite's time today.
- **Physics claims are checked by running the simulation,** never by
  reasoning alone.
- **The robot only knows what its parts could sense.** Nothing that steers
  the robot reads the simulator's truth. It reads its sensors, with the
  noise their datasheets give.
- **A comment in the code states a constraint that the code cannot show.**
  The story of how something was found goes in `docs/`, and the code keeps
  a one-line pointer to it. When a detail stops being true, its text is
  deleted, because git history keeps the record.
- **A change to what the live robots read or do starts a new period** in
  [docs/Observatory.md](docs/Observatory.md), because it changes what the
  live world's records mean.

The commands:

```bash
uv sync
uv run pytest -q -m "not slow"           # while you work: the fast tests
uv run pytest -q                         # before a change is done: the whole suite
uv run pytest -q --endurance-changed -m endurance   # and the flights it touches
uv run ruff check src/ scripts/ tests/   # the linter
```

The suite runs in parallel. It renders with EGL unless `MUJOCO_GL` names
another backend. A few long flights prove claims about physics, and they
run only when asked. The third command flies the ones whose paths your
change touches, which is usually none.

## Where people can help

The project is looking for people in three narrow roles first.

- **A task author for the benchmark.** The benchmark scores a robot on
  tasks that it has never seen. Those sealed tasks need an author who is
  not building the robot, so that the people who build it never see them.
- **A hardware reviewer.** Every part of the simulated robot is a real part
  that you could order, and nothing is ordered yet. Someone who has built
  robots can check the parts and the body against what real hardware does.
  [docs/Parts.md](docs/Parts.md) is the place to start.
- **A critic of the experiments' design.** The project's longer question is
  whether a stake in the world, feelings, a long memory and goals of its
  own make a robot better at learning new skills. Nothing about it is
  measured yet, and its design needs someone from outside to argue with it
  first.

If one of these roles fits you, open an issue that says which one, and what
you would bring to it. Questions and bug reports are welcome as issues too.

## License

PluggyBot is licensed under the [Apache License 2.0](LICENSE). Unless you
say otherwise, anything that you contribute is licensed under it too, as
section 5 of the license says.
