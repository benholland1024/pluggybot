# The webserver feature — telemetry out of the sim

How PluggyBot's world leaves the process: a recording on disk (v0) or a
live WebSocket stream (v1). Both speak the PluggyWorld wire protocol —
shapes, fixtures and the per-version changelog are `protocol/README.md`,
and nothing here restates them; the consumer side is
`rooftop-media-2026/docs/pluggyworld.md`. The stream exists so the deployed
world can be an observatory and so a visitor can witness, and talk to, the
robot (PluggyPlan.md "What this project is for"; Evaluation.md §5).

## The one seam, and the one rule

Every physics step in the hub stack bottoms out in `HubSwap._step_once`,
which fires `HubMission.step_hooks` — the same per-step callback list the
battery drains through. Everything here is another hook on that list, so it
works regardless of who owns the loop.

The rule all hooks obey: **no I/O inside the physics step**. A hook may
compare floats, build a dict, snapshot a small numpy array, and
`put_nowait` on a queue — serialization, PNG encoding, disk writes and
sockets all live on their own threads. The sim never waits for a consumer,
and a consumer that cannot keep up gets dropped messages, never a stalled
robot. Frames are due on SIM time, so a paused sim emits none (which is why
`mode` is also a heartbeat message — Overseer.md).

## The pieces (`src/pluggybot/telemetry/`)

- **`protocol.py`** — `PROTOCOL_VERSION`, the dynamic-body census both
  emitters share, and the two-repo vocabularies (visual hints, face states,
  message and inbound kinds).
- **`scene.py`** — MJCF→JSON scene transpiler: the static world, shipped
  once per world. `uv run python -m pluggybot.telemetry.scene
  [models/home_world.xml]` regenerates the fixture.
- **`recorder.py`** — `FrameBuilder`: decimation to `FRAME_HZ` = 20 Hz of
  sim time, sparse frames (a body ships only when it has moved more than
  `POS_EPS` = 0.5 mm since it was last *emitted*), a full keyframe every
  `KEYFRAME_S` = 5 sim-seconds marked `"key": true`; the sparse per-key
  blocks (activities, boards, screens, ledger, …) follow the same rule and
  re-ship on every keyframe. `TelemetryRecorder` is the builder plus a JSONL
  writer thread (`--record out.jsonl.gz` on `hub_lifecycle.py` / `serve.py`).
  `GridSampler` is the one grid implementation both sinks share: live at
  `GRID_HZ` = 1 Hz and never deduplicated (the relay hub caches the newest
  grid for late joiners, so silence would look like a broken path); a
  recording at 0.2 Hz, skipping an unchanged image, because those bytes are
  vendored into the website's bundle.
- **`pacer.py`** — `RealTimePacer`: a step hook that sleeps the headless
  loop so sim time tracks wall time at `--rate`. It only ever sleeps — when
  the sim falls behind (an osmesa render burst, a 1 ms-timestep swap) it
  stops sleeping until the sim catches up, so lag is transient drift, never
  a stall or a skip. `stats()` reports the drift; `resync()` after a pause
  stops a five-minute pause becoming a 2.9× sprint to catch up.
- **`publisher.py`** — `WsPublisher`: an outbound WebSocket **client** (the
  sim owns no public surface; if the endpoint is down the robot keeps
  living). Its own `FrameBuilder` instance, so live and recorded frames are
  identical. A bounded queue (`QUEUE_MAX` = 256, ~13 s of frames) feeds one
  sender thread, which also polls the socket for inbound visitor messages
  between sends (`mind/inbox.py`); every failure — endpoint down, socket
  death, slow consumer — degrades to dropped messages.
- **`vitals.py`** — `Watchdog`: the process's memory once a minute, and a
  runaway's stacks and allocations. Not on the wire; it writes to the log
  ("When the process dies", below).

**Two re-keying rules.** A sparse frame is deltas against what was last
*emitted*, so any gap leaves a consumer holding stale poses forever. (1) When
continuity breaks in our socket — a reconnect, a dropped frame — the
publisher resets the builder, re-sends the header and the next frame is a
keyframe (`test_reconnect_resends_header_and_keyframe`). (2) The far end is
the website's relay hub with browsers behind it; a browser joining
mid-mission never touches our socket, so keyframes also recur every
`KEYFRAME_S`, which bounds a joiner's wait to one interval and lets the hub
cache "last keyframe plus the frames since" without knowing anything about
bodies. Cost: 1 frame in 100.

**Not everything is a pose.** Ink, verdicts, deaths, narration and the map
are not per-tick quantities, so they ride the socket as typed messages
between frames (`draw`, `board_cleared`, `board_snapshot`, `earned`,
`grid`, `event`, `goals`, `death`, …); ink is never MuJoCo geometry — the
browser paints the polyline the pen actually inked. Three rules for a
consumer: dispatch on `type`, and no `type` means frame; a dropped typed
message is not superseded by a later one (a lost `draw` line is missing
until the board is next erased — the accepted price of never blocking
physics, and why recordings, not the live stream, are the lossless
artifact); and the `tasks` block is the one block that is not a delta —
present means complete, because a task can cease to exist. The full list,
each with its reason, is `protocol/README.md`.

**Everything on the socket is a readout.** Points are awarded by code, priced
by a data table and banked by a ledger that re-derives the payout; nothing
on the socket can move a balance, and a hidden-truth task's answer is
redacted from its `earned` message because the stream reaches the robot's
own context as well as the site (TaskPattern.md §2, §4).

**The ingest socket is authenticated.** `--token` (or `$PLUGGYWORLD_TOKEN`)
sends `Authorization: Bearer <token>` at the handshake. A refusal looks
exactly like a server that is down — a 1 s retry loop — so the publisher
keeps `last_error` and `serve.py` prints it when it never connected once;
a mis-deployed secret must not be a silent black hole.

## Running it

```
# terminal 1: any sink (the website's ingest socket in production)
uv run python scripts/ws_sink.py --port 8765

# terminal 2: the mission, live at 1x
MUJOCO_GL=osmesa uv run python scripts/serve.py --endpoint ws://localhost:8765

# …in the generated house + garden, which is what the site serves
MUJOCO_GL=osmesa uv run python scripts/serve.py --world home \
  --endpoint ws://localhost:8765

# …with the robot doing something: fetch the pen, erase a whiteboard, draw
# on it, stow the pen. `--boards` keeps what it drew across restarts.
MUJOCO_GL=osmesa uv run python scripts/serve.py --world home --errand draw \
  --boards var/boards.json --endpoint ws://localhost:8765

# …or both streamed surfaces in one run: draw, charge, then fetch the LCD
# and take a census of the garden. This is the queue the site's fixture is
# recorded from.
MUJOCO_GL=osmesa uv run python scripts/serve.py --world home --errand showcase \
  --boards var/boards.json --endpoint ws://localhost:8765

# …or rehearse the authenticated production path
uv run python scripts/ws_sink.py --port 8765 --token s3cret
PLUGGYWORLD_TOKEN=s3cret MUJOCO_GL=osmesa uv run python scripts/serve.py \
  --endpoint ws://localhost:8765/api/pluggyworld/ingest
```

`serve.py --pair` serves BOTH robots from one loop (issue #181; M12) under
`<world>_pair`: `--errand2` is the second robot's errand (default `none` —
it explores, then stands by for the shared board's work), `--robot-name-2`
its display name (default `Rowan`), and the documents live under
`--thoughts` (`<root>/` and `<root>/r2_pluggybot/`; `--goals`/`--journal`
are refused, they name one robot's files). One publisher, one board, one
ledger file with an account each, one operator switch on the primary.

`serve.py --rate 2.0` runs faster than life; `--free-run` disables pacing to
measure the machine's real-time multiple; `--record` keeps a v0 recording of
the same run; `--keyframe-s` tunes the keyframe cadence (0 disables, and late
joiners then wait forever); `--world {room_hub,home}` picks the world, and
picks it *whole* — model, scene name, rack pose, grid extent, battery, start
pose, errand destination and explore budget all come from
`lifecycle.world_config()`, since a half-applied world fails silently (a
short explore budget just stops filling the map; a stale errand destination
just drives at a wall). The arm, the pack, the mind and the state files are
flags too (`--help`; CLAUDE.md). `ws_sink.py` measures received frame *gaps*
— the wall-clock spacing between frames — which is the consumer-side proof
the stream is smooth, reports keyframe spacing, which is the proof a late
joiner converges, and talks back (type a line: it goes down the socket as a
visitor message).

## Deploying it (rooftop-media-2026 #20)

```
docker build --build-arg PLUGGY_COMMIT=$(git rev-parse --short HEAD) \
  -t pluggyworld-sim .                     # from the repo root
docker run --rm -e PLUGGY_ENDPOINT=ws://host.docker.internal:8765 \
  pluggyworld-sim                          # against a local ws_sink.py
```

The image (`Dockerfile`, `deploy/`) runs `serve.py` and nothing else. Five
things about it are decisions rather than boilerplate:

- **It is not the dev environment.** The serve path imports mujoco, numpy,
  scipy, pillow, websockets, the apriltag detector and the overseer's client
  — no torch, no ultralytics, no SB3, which would put a ~3 GB CUDA wheel on a
  machine with no GPU. The image installs `deploy/requirements-serve.txt`,
  pinned to `uv.lock`, and `tests/test_deploy.py` fails if the pins drift
  from the lock or the mission stack grows an import the image omits — it
  blocks the omitted packages and actually flies the robot, because the
  detector is a lazy import no import scan can see. Both failure modes are
  otherwise *silent*: green suite here, dead container there.
- **osmesa, and only osmesa.** `libosmesa6` is the whole GL story;
  `MUJOCO_GL=osmesa` is baked in, and the Dockerfile renders one offscreen
  frame at BUILD time, so "headless GL works on this machine" is answered by
  `docker build` rather than by a mission that falls over ten minutes in.
- **The commit is baked, and the build is red without it.** `.git` is
  dockerignored, so `PLUGGY_COMMIT` is a required build arg and the stream's
  header carries it (Evaluation.md §5: observatory data without a build
  identity is unusable data). Reading it back — the commit, the documents,
  every decision, event and memory write — is one `curl` against the site's
  `GET /api/pluggyworld/observe` with a read token; Evaluation.md §5 "How the
  observatory is read" is the record and the worked example.
- **No ports, no `depends_on`.** The sim is an outbound client that retries
  every second, so it needs no inbound rule and no place in the reverse
  proxy, and it survives the website being restarted underneath it.
- **Config is environment, not a command line.** `deploy/entrypoint.sh`
  turns `PLUGGY_*` variables into flags and is the list; the data files
  (`$PLUGGY_REWARDS`, `$PLUGGY_CADENCE`, …) and the mind's settings are read
  by the sim directly. The ingest secret stays `$PLUGGYWORLD_TOKEN` and the
  API keys stay in the environment, never flags, because a flag is visible in
  `ps`. Anything passed to the container is appended after the derived
  flags, so `docker run <image> --rate 2.0` still wins.

**This repo owns the image; the website repo owns the deployment.** The
`sim:` service lives in `rooftop-media-2026/compose.yaml` — one copy, not a
snippet here that drifts from it. It builds from `context: ../pluggybot`
(the two repos side by side) behind a `sim` compose profile, so `docker
compose up` still works for someone who only has the website. What it
encodes that matters from this side: `/var/lib/pluggybot` is a named volume
because boards, the ledger, the task board, the journal and the thought
files are **world** state, and `restart: unless-stopped` starts the next
process when one ends — which since #345 carries on from where the last one
stopped (below) rather than from the start pose. It serves `--pack hosting` (8 Wh on home) rather than the demo
cell, which flattens in minutes; the low-battery reserve is deliberately
*not* scaled with the pack — it is the absolute energy needed to reach the
dock, a property of the floor plan. Which arm the served world flies is
`$PLUGGY_ARM` / `$PLUGGY_ORIGIN`: `autonomous`, both robots, `unseeded`
(issue #206; the argument is Evaluation.md §2).

## A restart is a continuation (issue #345)

The served process ends for four reasons: a deploy, a crash, an out-of-memory
kill, and the `PLUGGY_MAX_SIM_TIME` ceiling (3600 sim s in compose, so every
hour). Until #345 each new process built the world from its XML: both robots
at their spawn poses on a full pack, both maps empty, every module back on
its bay, and the job in hand failed. The volume kept what the robots had
WRITTEN and nothing they had done, so the sixth quality's buffer was refilled
for free every hour, and the robots planned round it ("woke up in home with
the pack at 100%").

`$PLUGGY_WORLD_STATE` (`/var/lib/pluggybot/world.npz` in the image) holds the
rest, and `src/pluggybot/continuation.py` keeps it:

- **What is saved.** For the physics: every joint's position and velocity,
  the solver's warm start and every actuator's control, matched by NAME, plus
  the mocap mouse and the sim clock. Per robot: its pack, its believed pose,
  the rack belief with its sightings, the occupancy grid and the height map,
  and the lidar's and depth camera's noise generators. Also whether it is
  dead and since when, its survival and unminded clocks, its explore state,
  the errand it was in, and what it had queued. For the world: its
  activities (the mouse, the plates, the pair's encounters) and the
  producer's schedule. The ledger, boards, task board, thoughts and tickets
  were already files and are not saved twice.
- **When.** Every `SAVE_EVERY_S` (60 sim s) on the last robot's step
  hooks, so a pair is saved after both robots' step, and once more when the
  run ends. A run ends at its budget or on
  SIGTERM/SIGINT; a signal only asks, and the next step boundary ends the
  day before the save. A crash is never saved: the next process carries on
  from the last minute's save. The ledger, the boards and the task board
  are written on every event and are not rewound, so a job paid in that
  minute stays paid.
- **Put back.** `continuation.load` finds the save before the task board is
  built. Each lifecycle's `begin()` re-hangs its built tools, then
  `continuation.restore` puts the bodies back. The day routine opens without
  moving: no start pose, no spin. History says "the world restarted; I
  carried on from (x, y) with the pack at N%".
- **Sim time continues.** Every absolute stamp keeps its meaning: the
  survival clock, the unminded clock, a dead robot's stand-up timer, an
  offer's deadline (the board loads without rebasing). `max_sim_time` is
  the RUN's budget from where it starts. The wire's `t` grows past 3600,
  which the site handles, since every new header resets its view.
- **The errand in flight ends**, because it was a generator. Its job stays
  the robot's: an errand job is queued again (rebuilt off the task, with its
  committed answer), and a procedure job stays claimed. A module the restart
  left on the fork is stowed first. A claim held by a robot not in the new
  world goes back on offer. A game still fails, because its referee lived in
  the process. A job taken up through `MAX_TAKE_UPS` (3) restarts without
  finishing is failed: the world's crash-loop guard counts saves, and a job
  whose errand crashes the process would otherwise crash every process after
  it.
- **An offer sets its props out.** The tower's blocks and the bench's cubes
  go back where the world compiled them when their challenge is offered,
  except one touching a robot. The hourly reset used to be what made "set
  out in a row at …" true; with the world carried on, a failed attempt would
  otherwise leave them wherever it dropped them, for good.
- **Two refusals.** A world whose geometry changed (the `fingerprint` over
  bodies, joints and geoms, taken before any built tool is hung) gets its
  clock, packs, deaths and jobs, but not its bodies or maps: the robots
  start from their start poses and are told why. And a save restored
  `MAX_RESUMES` (3) times with no new save in between is not trusted again:
  the next start is fresh, and History says why. A file that cannot be read
  at all (empty, torn) is a fresh start too, never a crash. A grid that does
  not fit the build's is left empty and the robot explores again, where it
  stands.
- **Parity.** `scripts/determinism_spike.py --resume-at T` flies a scripted
  day straight through, then the same day saved at the first idle pass of
  the loop past T and carried on in a new process. After the restore the two
  are IDENTICAL: 761 state samples on room_hub, and 2404 over 1202 s of the
  home world's day (four jobs, two drawings, two censuses, the cage and the
  plates, a 503 s charge). The first check caught two
  defects. The sensors' noise generators were re-seeded, so the first scan
  painted a different map and the route parted 3 s later. And
  `Task.from_json` re-priced an open offer at its kind's generic figure: a
  room_hub carry went from 0.817 to 0.93 Wh after every restart, too dear
  for a pack at 88 %.
- **Cost.** About 50 ms of the physics thread per save on the home pair with
  both maps built, 1.3 MB on disk (zlib level 1; the default level 6 cost
  175 ms).

Not kept: a decision in flight, the mind's in-process context (it reads
History), a visitor message still in the inbox, and an open `look`.

## When the process dies (issue #349)

A process can end without a word. A SIGKILL leaves no traceback and no
summary: the kernel's OOM killer sends one at the container's 2 GiB
`mem_limit`. So does a fatal signal inside MuJoCo or osmesa. Between
2026-09-24 11:47 and 2026-09-25 04:00 UTC seven served processes stopped
that way: every run on the observatory's `runs` list that ended before its
hour, three on `42f4a11` and four on `a803c83`. Two of them were sampled
once a minute (`docker stats`). Each ran away at 115–120 MiB a minute from
about 1 GiB, with neither robot deciding anything, and died at 1.89 and
1.95 GiB. One of them had grown ~2.7 MiB a minute at rest for 39 minutes
before that.

So `serve.py`'s `main()` runs a watchdog (`telemetry/vitals.py`), and the
log carries:

- **`vitals: rss 843 MiB (+0.3 MiB/min), peak 843 MiB, t=1234.5 s`** once a
  wall minute: resident memory, its rate, and the sim clock. A runaway
  that has stopped the world shows as `t` standing still.
- **`vitals: RUNAWAY -- ...`** when the rate has run over 50 MiB a minute
  for two samples in a row. The first three samples are a warm-up (the
  build and the carry-on take the process from 212 to 788 MiB in its first
  minute). Every thread's stack follows (`faulthandler`, with thread
  names), and `tracemalloc` starts. 20 s later come the allocations made
  since and still held: the twelve largest, eight frames each, newest
  frame first. Then the stacks again. This happens **once per episode**:
  another report needs two calm minutes first.
- **`Fatal Python error: ...`** and every thread's stack on a segfault or
  an abort.
- **`vitals: exiting -- <why>`** when the process ends by any path Python
  sees: the run ended, an exception (named), a second signal. **A log that
  ends without it was a kill.** If the last `vitals: rss` lines were
  climbing, the kill was at the memory cap.

Read it off the box: `ssh netcup docker logs rooftop-prod-sim-1 2>&1 | grep
-A250 'vitals: RUNAWAY'`. The log covers the current container only, so a
deploy loses it: read it before rebuilding.

Why the numbers are what they are:

- **50 MiB a minute, not 30.** At rest, the busiest minutes on record read
  +23 and +20 back to back (`3518953`, 2026-09-25). A threshold of 30
  clears them by 1.3×. 50 sits about 2× from them and 2× from the
  runaway's 115.
- **Tracing starts at the onset, never at boot.** MEASURED on 2026-09-25,
  with the pair free-running on the dev machine (EGL), four interleaved
  runs, per wall minute after start-up: 45 sim-seconds untraced, and 7.8
  traced from boot with eight frames. That is 5.8× slower. On a box where
  the pair only just holds 1×, tracing from boot would leave no served
  world. A runaway that is still growing grows in whatever it traces from
  the onset on.
- **20 s of tracing, and no filter.** The pair runs ~6× slower while it
  traces, and the pacer catches up afterwards. Building the report is ~3 s
  of Python per million live traces, taken from the physics thread's share
  of the GIL. `Snapshot.filter_traces` cost another 11 s per million. At
  the measured runaway, 20 s is at most ~38 MiB.

What it cannot see: memory that C allocates for itself (MuJoCo, osmesa, a
C extension's own `malloc`) is not traced; numpy's arrays are. A report
whose traced total is small against the `rss` growth is that answer, and
the stacks still say what every thread was doing.

The hourly ceiling (`PLUGGY_MAX_SIM_TIME`, #345) is decided afterwards,
off these lines, once the fix for what they find is deployed.

## Measured

The constraint the Dockerfile relies on: **osmesa software rendering carries
the served world above 1× real time on four cores, and four dedicated cores
is the floor, not a comfortable choice.** Measured on the dev machine
(2026-08-16, `--world home --errand draw`, `MUJOCO_GL=osmesa --free-run`,
`taskset -c 0-3`): **1.07× real time** (308.3 s sim / 287.1 s wall),
6095 frames, **0 dropped**, peak RSS 621 MB, ~1.9 cores busy — a 7 %
margin, where `room_hub` had shown 30 %. A paced 1× `room_hub` run held drift
to −0.25 s over 178 s (0.14 %); its worst transient lag (~1 s) and worst
received frame gap (~0.6 s) both sit at the tool swap, where the timestep
drops to 1 ms and step cost doubles, and the pacer absorbs them with no
frames lost. Keyframes are 1 % of frames and 2.6 % of bytes.

⚠ These are dev-machine numbers on a world that has since grown (the
expanded house, more errands, an overseer): re-measure with `--free-run`
before trusting the margin, on the box that will serve, with nothing else
running — a full-suite `pytest` starting on the same box once turned a paced
1× run into 0.71× and −123 s of drift.

**A pair costs 2.15× one robot** (2026-09-13, dev machine, EGL, all cores,
`--world home --pack hosting --tasks --metabolism --free-run`, 180 s
budget): one robot **1.25×** real time (321.8 s sim / 258.4 s wall), the
pair **0.58×** (274.7 s / 473.9 s), 5 420 frames, 0 dropped. Against the
deploy box's 1.07× for one robot on four pinned cores, a pair there lands
near 0.5×: serve it at `PLUGGY_RATE=0.5`, or give the service ~8 cores and
re-measure with `--pair --free-run` before trusting 1×.

**The served pair ran at 0.23×, and it was the shadows** (rooftop-media-2026
#296, 2026-09-19; SimNotes has the story). On the deploy box under osmesa a
tag-camera frame cost 1113 ms with the home world's sixteen shadow-casting
lights and 32 ms without; the detector renders without shadows now, and
the same pair measured **0.54×** on the box (53.6 s sim / 100 s wall, four
cores, the production sim contending) with the container on one core --
the physics thread. **Then the thread was profiled** (SimNotes,
"four-fifths bookkeeping"): half of it was Python walking the contact
list struct by struct every step, and a tenth the map's inflation. With
those read as arrays the same pair measured **0.96×** over the 60 s carry
(53.6 s sim / 56 s wall) and **0.80×** over its whole day (259 s / 324 s,
the swaps at 1 ms timesteps being the dear part), 2026-09-20. A pair is
one physics thread: more cores do not move it, and the site's
pace-following clock covers what is left.
