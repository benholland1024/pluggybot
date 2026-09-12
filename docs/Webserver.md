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
mission when one ends — each restart is a fresh mission from the start pose
("woke up at home"): the volume persists, the map, the battery level and the
pose do not. It serves `--pack hosting` (8 Wh on home) rather than the demo
cell, which flattens in minutes; the low-battery reserve is deliberately
*not* scaled with the pack — it is the absolute energy needed to reach the
dock, a property of the floor plan. Which arm the served world flies is
`$PLUGGY_ARM`, and it stays `guarded` (Evaluation.md §2).

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
1× run into 0.71× and −123 s of drift. A second robot in the shared world
needs this measured again rather than assumed.
