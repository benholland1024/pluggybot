# The webserver feature — telemetry out of the sim

How PluggyBot's world leaves the process: a recording on disk (v0) or a
live WebSocket stream (v1). Both speak the PluggyWorld wire protocol —
shapes and fixtures in `protocol/README.md`, consumer architecture in the
design doc (`rooftop-media-2026/docs/pluggyworld.md`).

## The one seam, and the one rule

Every physics step in the hub stack bottoms out in `HubSwap._step_once`,
which fires `HubMission.step_hooks` — the same per-step callback list the
battery drains through. Everything here is just another hook on that list,
which is why it works identically under the blocking lifecycle today and
will keep working after the tick refactor: the hooks fire regardless of
who owns the loop.

The rule all hooks obey: **no I/O inside the physics step**. A hook may
compare floats, build a dict, snapshot a small numpy array, and
`put_nowait` on a queue — serialization, PNG encoding, disk writes, and
sockets all live on their own threads. The sim never waits for a consumer.

## The pieces (`src/pluggybot/telemetry/`)

- **`protocol.py`** — `PROTOCOL_VERSION` + the dynamic-body census both
  emitters share.
- **`scene.py`** — MJCF→JSON scene transpiler (v0): the static world,
  shipped once. `uv run python -m pluggybot.telemetry.scene` regenerates
  `protocol/scene.room_hub.json`.
- **`recorder.py`** — `FrameBuilder` (decimation to ~20 Hz of sim time +
  sparse frames: a body ships only when it has moved > 0.5 mm since last
  emitted, with a full keyframe every `KEYFRAME_S` = 5 sim-seconds) and
  `TelemetryRecorder` (v0: builder + a JSONL writer thread).
  `scripts/hub_lifecycle.py --record out.jsonl.gz` produces a replayable
  full-mission recording.
- **`pacer.py`** — `RealTimePacer` (v1): a step hook that sleeps the
  headless loop so sim time tracks wall time at a configurable rate.
  It only ever sleeps — when the sim falls behind (an osmesa render
  burst, a 1 ms-timestep swap) it simply stops sleeping until the sim
  catches back up, so lag is transient drift, never a stall. `stats()`
  reports the drift so pacing accuracy is measured, not assumed.
- **`publisher.py`** — `WsPublisher` (v1): an outbound WebSocket
  **client** (the sim owns no public surface — if the endpoint is down,
  the robot keeps living). Same `FrameBuilder` code as the recorder (its
  own instance), so live and recorded frames are identical. A bounded queue feeds a sender
  thread; every failure mode — endpoint down, socket death, slow
  consumer — degrades to dropped messages, never to blocked physics.
  Two extra message types ride the socket: the occupancy-grid belief as
  a base64 PNG at ~1 Hz, and lifecycle narration lines as events.

**Sparse frames + an unreliable pipe need re-keying.** A frame is deltas
against what was previously *emitted*, so any gap — a reconnect, a
dropped frame — leaves the consumer holding stale poses forever. Whenever
continuity breaks, the publisher resets the builder and the next frame is
a full keyframe; every new connection also re-opens with the stream
header. `tests/test_webserver.py` guards this (the reconnect test fails
without the reset).

**…and the consumer we cannot see needs re-keying on a timer.** That
recovery only covers breaks in *our* socket. In production the far end is
the website's relay hub (rooftop-media-2026 #22) with browsers behind it:
a browser joining mid-mission never touches our connection, so no
reconnect fires and nothing re-keys for it. Everything that had stopped
moving — the rack, a stowed module, a parked robot — is simply absent from
its world, permanently. So keyframes also **recur** every 5 sim-seconds
and are marked `"key": true`, which bounds a joiner's wait to one interval
and lets the hub cache exactly "the last keyframe plus the frames since"
without knowing anything about bodies. Cost: 1 frame in 100. That is a
protocol shape change — hence `protocolVersion` 0.2.0 and re-vendored
fixtures.

**Activities ride the same rails** (`protocolVersion` 0.3.0, issue #8).
Frames may carry an `activities` block — the task state machines' discrete
world state — sparse exactly as poses are, and re-shipped on every keyframe.
That last part matters more here than for poses: an activity's visible
effect usually lives on a **static** body (the reference gate is a mocap
body that ships once in the scene description and never again), so for a
change like that the flag is the *only* record anywhere in the stream. In
the measured home mission the block costs **1.0 % of frames**. Another
shape change, so another re-vendor; see `docs/ActivityPattern.md`.

**Drawings ride it as events, not as poses** (`protocolVersion` 0.4.0,
issue #12). A stroke is not a per-tick quantity — decimating one to 20 Hz
would mean shipping the same polyline a hundred times or dropping it — so
each finished stroke goes out as its own `draw` message (board id plus the
polyline the pen *actually inked*), and erasing a board as `board_cleared`.
The browser paints them into a canvas texture; ink is never MuJoCo
geometry. Frames additionally carry a sparse `boards` block — programs,
strokes, fill, last cleared — on exactly the activities rule, and for
exactly the activities reason: a drawing has no body, so nothing about it
appears in the pose stream at all.

Two consequences worth knowing before writing a consumer. **Recordings are
now a mixed stream**: dispatch on `type`, and no `type` means frame. And a
dropped `draw` is unlike a dropped frame — no later message supersedes it,
so that line is missing from the canvas until the board is next erased.
That is the accepted price of never blocking physics on a socket, and it is
why recordings, not the live stream, are the lossless artifact.

**A face and a catch-up channel** (`protocolVersion` 0.5.0, issue #13).
Frames gain a sparse `screens` block — what each display module is showing,
`{mode, powered, face, hint}` — on the same rule again, and for the same
reason a third time: an LCD's content is not a pose. The sim streams an
ENUM and the browser draws the face, which is what makes an expressive
robot cost five short strings per change instead of a texture.

And a third typed message, **`board_snapshot`**: every stroke a board is
currently carrying, sent right after the header on every connect. It closes
the one hole the mixed stream left. A keyframe re-ships each board's
*counters* but never its *lines* — a `draw` happens once — so a browser
that joins mid-mission, or a sim restarting onto its saved board state, had
no way to learn what was already on the wall. `--boards` state now keeps the
polylines themselves for exactly this. A relay hub should cache the latest
snapshot per board plus the strokes since it, and drop both on a
`board_cleared`.

### 0.6.0 — the robot is scored

Frames gain a sparse `ledger` block (balance, totals, the last few earnings)
and a fourth typed message, **`earned`**, carrying one finished task's verdict
as the sim banks it. Same sparse + keyframe rule as the three blocks before it,
with one difference that saves the relay hub a cache: `recent` rides in every
keyframe, so a late joiner needs no snapshot message for points the way it
needs one for ink.

Everything about it is a READOUT. Points are awarded by a deterministic
evaluator in the sim (`economy/scoring.py`), priced by a data table
(`economy/rewards.json`), and banked by a ledger that re-derives the payout before
accepting it. Nothing on the socket can move a balance, in either direction —
including, when it lands, the LLM overseer, which sees its score and cannot
touch it. `--ledger PATH` (or `$PLUGGY_LEDGER`) is where the balance lives
between runs, exactly as `--boards` is for ink.

`--tasks` / `--task-state PATH` (or `$PLUGGY_TASKS`, which implies both) turn
on the **job board** (issue #21): work the house or a visitor puts up, which
the robot may take, and which is graded and paid through exactly the machinery
above — a task names an evaluator and a reward-table row and carries no payout
of its own, so nothing that can create a job can price one. Off by default,
because a task board adds errands and reshuffles a whole mission.

⚠ **A hidden-truth task publishes its verdict without its answer.** The
census's real count is redacted from the message's `metrics` and its `reason`,
because this stream reaches both the website and the robot's own context.

**The ingest socket is authenticated.** `--token` (or `$PLUGGYWORLD_TOKEN`)
sends `Authorization: Bearer <token>` at the handshake. A refusal looks
exactly like a server that is down — a 1 s retry loop — so the publisher
keeps `last_error` and `serve.py` prints it when it never connected once;
a mis-deployed secret should not be a silent black hole.

## Running it

```
# terminal 1: any sink (the website's ingest socket in production)
uv run python scripts/ws_sink.py --port 8765

# terminal 2: the mission, live at 1x
MUJOCO_GL=osmesa uv run python scripts/serve.py --endpoint ws://localhost:8765

# …in the generated house + garden, which is what the site serves (issue #9)
MUJOCO_GL=osmesa uv run python scripts/serve.py --world home \
  --endpoint ws://localhost:8765

# …with the robot actually doing something: fetch the pen, erase a
# whiteboard, draw on it, stow the pen (issue #12). `--boards` keeps what it
# drew across restarts; `--errand draw2` does two boards with a charge in
# between.
MUJOCO_GL=osmesa uv run python scripts/serve.py --world home --errand draw \
  --boards var/boards.json --endpoint ws://localhost:8765

# …or both streamed surfaces in one run (issue #13): draw a house, charge,
# then fetch the LCD and take a census of the garden, reporting the count on
# the robot's screen. This is the queue the site's fixture is recorded from.
MUJOCO_GL=osmesa uv run python scripts/serve.py --world home --errand showcase \
  --boards var/boards.json --endpoint ws://localhost:8765

# …or rehearse the authenticated production path
uv run python scripts/ws_sink.py --port 8765 --token s3cret
PLUGGYWORLD_TOKEN=s3cret MUJOCO_GL=osmesa uv run python scripts/serve.py \
  --endpoint ws://localhost:8765/api/pluggyworld/ingest
```

`serve.py --rate 2.0` runs faster than life; `--free-run` disables pacing
to measure the machine's real-time multiple; `--record` keeps a v0
recording of the same run; `--keyframe-s` tunes the keyframe cadence;
`--world {room_hub,home}` picks the world, and picks it *whole* — model,
scene name, rack pose, grid extent, battery size, start pose, errand
destination and explore budget all come from `world_config()`, since a
half-applied world fails silently (a short explore budget just stops
filling the map; a stale errand destination just drives at a wall).
`ws_sink.py` measures received frame *gaps* — the wall-clock spacing
between frames — which is the consumer-side proof the stream is smooth,
and reports keyframe spacing, which is the proof a late joiner converges.

## Deploying it (rooftop-media-2026 #20)

```
docker build -t pluggyworld-sim .          # from the repo root
docker run --rm -e PLUGGY_ENDPOINT=ws://host.docker.internal:8765 \
  pluggyworld-sim                          # against a local ws_sink.py
```

The image (`Dockerfile`, `deploy/`) runs `serve.py` and nothing else. Four
things about it are decisions rather than boilerplate:

- **It is not the dev environment.** The serve path imports mujoco, numpy,
  scipy, pillow and websockets — no torch, no ultralytics, no SB3. Those
  belong to training, dataset generation and the detector, none of which
  run on a serving box, and installing them would put a ~3 GB CUDA wheel
  on a machine with no GPU. So the image installs
  `deploy/requirements-serve.txt`, pinned to `uv.lock`, and
  `tests/test_deploy.py` fails if either the pins drift from the lock or
  the mission stack grows an import the image omits. Both of those
  otherwise fail *silently* — green suite here, dead container there.
- **osmesa, and only osmesa.** `libosmesa6` is the whole GL story;
  `MUJOCO_GL=osmesa` is baked into the image. The Dockerfile renders one
  offscreen frame at BUILD time, so "headless GL works on this machine" is
  answered by `docker build` on the server rather than by a mission that
  falls over ten minutes in.
- **No ports, no `depends_on`.** The sim is an outbound WebSocket client
  that retries every second, so it needs no inbound rule and no place in
  the reverse proxy, and it survives the website being restarted or
  redeployed underneath it.
- **Config is environment, not a command line** (`deploy/entrypoint.sh`):
  `PLUGGY_ENDPOINT`, `PLUGGY_WORLD`, `PLUGGY_ERRAND`, `PLUGGY_RATE`,
  `PLUGGY_BATTERY_WH`, `PLUGGY_MAX_SIM_TIME`, `PLUGGY_BOARDS`,
  `PLUGGY_LEDGER`, `PLUGGY_TASKS`. `PLUGGY_ROBOT_NAME` (issue #39) is read
  by the frame builder itself rather than the entrypoint — the robot's
  display name on the wire, `"Pluggy"` unless a deployment names its robot
  (`--robot-name` overrides; the robot *id* stays the body name either
  way, so renaming re-keys nothing). The ingest
  secret stays `$PLUGGYWORLD_TOKEN`, read by `serve.py` itself, because a
  flag is visible in `ps`. Anything passed to the container is appended
  after the derived flags, so `docker run <image> --rate 2.0` still wins.

**This repo owns the image; the website repo owns the deployment.** The
`sim:` service lives in `rooftop-media-2026/compose.yaml` — one copy, not a
snippet here that drifts from it. It builds from `context: ../pluggybot`
(the two repos checked out side by side) and sits behind a `sim` compose
profile, so `docker compose up` still works for someone who only has the
website. Two things it encodes that are worth knowing from this side:
`/var/lib/pluggybot` is a named volume because board contents are **world**
state — a mission ends when its errands do, `restart: unless-stopped`
starts the next one, and the volume is what makes the robot walk into the
house it left rather than a blank one. And `PLUGGY_BATTERY_WH` is 8.0
rather than the world's 1.1 Wh demo cell, which flattens in minutes: a
watched world wants hours between charges. The low-battery reserve is
deliberately *not* scaled with it — it is the absolute energy needed to
reach the dock, not a fraction of capacity.

What the image does **not** do is keep one continuous world alive: each
restart is a fresh mission from the start pose, the "woke up at home" model
the design doc allows. Boards persist; the map, the battery level and the
pose do not. A standing world is the tick refactor's job, not this issue's.

## Measured (dev machine, 2026-08-14; protocol 0.1.0)

Full battery-driven `room_hub` mission — explore → errand (pick, carry,
stow) → battery to 0 % as the pins connect → charge to 90 % — 178.1 sim-
seconds, 3500 frames + 175 grid images + 8 events streamed to a local
`ws_sink.py`, **zero frames dropped** in every run:

| Configuration | Result |
|---|---|
| `MUJOCO_GL=osmesa`, paced `--rate 1.0` | 178.1 s sim / 178.3 s wall (**1.00×**); drift −0.25 s at close, worst transient lag 1.03 s; received frame gaps median 41 ms, p99 152 ms, max 616 ms |
| `MUJOCO_GL=osmesa`, `--free-run`, `taskset -c 0-3` (the 4-core CPU-server shape) | **1.30× real time** (178.1 s sim / 137.4 s wall) |
| `MUJOCO_GL=egl`, `--rate 3.0` (deliberately beyond the machine) | sustains 1.80×; pacer reports the shortfall honestly (drift −21.6 s) rather than stalling or skipping sim time |

Reading the numbers: osmesa software rendering keeps a paced 1× live
stream with ~30 % CPU headroom on four cores — the Phase-0 question from
the design doc, answered yes for this machine (a dedicated-core server
still needs its own run). The transient ~1 s lag and the ~0.6 s worst
frame gap both occur around the tool swap, where the timestep drops to
1 ms and step cost doubles; the pacer absorbs it by not sleeping until
the sim catches back up, and the stream recovers within a second with no
frames lost. Pacing accuracy over the mission is −0.25 s in 178 s
(≈0.14 %).

### Re-checked on 0.2.0 (2026-08-15)

Recurring keyframes changed the payload, so the stream was re-measured —
`MUJOCO_GL=osmesa --free-run`, but **not** core-pinned, so it is not
comparable to the 1.30× row above: 176.8 s sim / 121.4 s wall (1.46×),
3475 frames + 173 grids + 8 events, **zero dropped**, received frame gaps
median 26 ms / p99 108 ms / max 230 ms, and 36 keyframes spaced 5.00–5.05
sim-seconds apart (the cadence re-anchors on the frame that carried the
keyframe, so it runs at most one frame interval late — never early).

The added payload is small: keyframes are 1.0 % of frames and 2.6 % of
bytes, and the committed recording grew 224,550 → 227,499 B (+1.3 %)
gzipped. The mission itself is now 176.8 s rather than 178.1 s — that is
the home-world commit's battery re-tune and navigation fixes, not the
protocol, and it means the recording fixture had been stale against the
committed code until this regeneration. Unlike the scene fixture, nothing
tests the recording against current behaviour (re-running a full mission
inside pytest is too expensive), so **regenerate it whenever mission
behaviour changes**, not only when the protocol does.

### The world the site actually serves (2026-08-16; protocol 0.4.0)

Every number above is `room_hub` carrying an LCD. The site serves the
house, drawing (issue #9, #12), and that is a heavier world: more geometry
for the lidar and the tag cameras, a plotting errand, 308 s of mission
instead of 178, and 6095 frames instead of 3500. Measured the same way —
`MUJOCO_GL=osmesa --free-run`, `taskset -c 0-3`, `--world home --errand
draw`:

| Configuration | Result |
|---|---|
| `--free-run`, `taskset -c 0-3` | **1.07× real time** (308.3 s sim / 287.1 s wall), 6095 frames, **0 dropped**, peak RSS 621 MB, ~1.9 cores busy |

So the answer to the Phase-0 question is still yes, but the margin is 7 %,
not the 30 % `room_hub` showed. Read that as **four dedicated cores is the
floor for the served world, not a comfortable choice** — the shared-vCPU
budget probe from the design doc's ladder is off the table, and a second
robot in the shared world will need this measured again rather than
assumed.

**Paced 1× could not be validated on this machine, and the reason is
worth writing down.** The paced run came back at 0.71× with −123 s of
drift, which contradicts a 1.07× free-run capability. The obvious suspect
is the pacer's own sleeping, and it is not: 1000 × 20 ms sleeps under the
same load overshot by 0.200 ms each (worst 8.8 ms), which is ~3 s over a
whole mission, not 123. What actually happened is that a full-mission
`pytest` run started on the box at the same moment and took roughly a core
for the duration. This is a *dev-machine* result, not a property of the
sim — and it is exactly the stutter the design doc predicts for shared
cores. The decisive run is the one on the server, which is also the
issue's third acceptance box.
