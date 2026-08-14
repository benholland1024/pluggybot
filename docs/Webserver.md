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

# …or rehearse the authenticated production path
uv run python scripts/ws_sink.py --port 8765 --token s3cret
PLUGGYWORLD_TOKEN=s3cret MUJOCO_GL=osmesa uv run python scripts/serve.py \
  --endpoint ws://localhost:8765/api/pluggyworld/ingest
```

`serve.py --rate 2.0` runs faster than life; `--free-run` disables pacing
to measure the machine's real-time multiple; `--record` keeps a v0
recording of the same run; `--keyframe-s` tunes the keyframe cadence.
`ws_sink.py` measures received frame *gaps* — the wall-clock spacing
between frames — which is the consumer-side proof the stream is smooth,
and reports keyframe spacing, which is the proof a late joiner converges.

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
