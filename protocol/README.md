# The PluggyWorld wire protocol — canonical fixtures

This directory is the data contract between pluggybot (the producer) and
the `rooftop-media-2026` website (the consumer). The website repo vendors
copies of these files into its test fixtures and never imports pluggybot
code; both repos' tests run against the same recorded artifacts. Design
doc: `rooftop-media-2026/docs/pluggyworld.md`, § "The scene protocol" and
§ "Repo topology"; the website-side spec lives with its protocol issue.

**Versioning.** Every artifact carries `protocolVersion`
(`pluggybot.telemetry.protocol.PROTOCOL_VERSION`, currently `0.4.0`).
Bumping it is a deliberate two-repo event: change the shape, bump the
version, regenerate these fixtures, and re-vendor them in the website repo.
`tests/test_telemetry.py` fails if the committed fixtures drift from the
committed world or the committed version.

### 0.3.0 → 0.4.0 (whiteboards are state, and ink is an event)

Boards became persistent world state the sim owns and streams (issue #12).
**Mostly additive, with one thing to fix in a replayer** — see the second
bullet.

- Frames may carry a **`boards`** object: per drawing surface, which stroke
  programs are on it, how much of the pen's reach carries ink, and when it
  was last cleared. Sparse and keyframe-refreshed on exactly the same rule
  as `activities`. The header gains **`"boards": [names]`**.
- **Recordings are now a mixed stream.** Two typed messages ride *between*
  the frames — `draw` and `board_cleared` (below) — where before, every
  line after the header was a frame. **Dispatch on `type`; no `type` means
  frame.** That rule already governed the live stream; it is now true of
  `.jsonl` recordings too, and a 0.3.0 replayer that assumed otherwise
  trips over the new lines.

```jsonc
// one stroke, emitted as the pen finishes it
{"type": "draw", "t": 123.4, "robot": "pluggybot", "board": "whiteboard_a",
 "program": "house", "stroke": 3,
 "points": [[lat, height], ...]}     // board-local metres, +lat = viewer's LEFT

// the board was erased (start of a drawing errand, or an explicit action)
{"type": "board_cleared", "t": 120.0, "robot": "pluggybot",
 "board": "whiteboard_a", "clearedAt": "2026-08-16T08:01:09+00:00"}
```

**Strokes are never geometry.** Ink is layer 3 of the three-layer model
(rigid bodies in MuJoCo · discrete state in Python · everything organic in
the browser): the site paints these polylines into a canvas texture on the
board mesh. Nothing about a drawing appears in the pose stream, so these
messages plus the `boards` block are the *only* record of it — the same
argument that put `activities` on the wire in 0.3.0.

**Consumer notes.** `points` are in the board's own frame, origin at the
board's centre, metres, rounded to 0.1 mm — the board's pose and half-extents
come from the scene description. `lat` is measured to the LEFT of the robot's
approach, so a canvas drawn in screen coordinates flips it once. The polyline
is what the pen actually **inked**, not what it was commanded, so it carries
the robot's real ~1 mm of form error and a stroke that lost the board comes
through short. `board_cleared` means *drop everything you have painted for
that board*; a `draw` event whose board you have no canvas for is safe to
ignore. Board `fill` is measured against the window the pen can reach with
the base parked (110 × 200 mm on the home boards), not against the whole
slab.

### 0.2.0 → 0.3.0 (activities join the frame)

Additive; a 0.2.0 consumer ignores the new block and needs no changes.

- Frames may carry an **`activities`** object: the task state machines'
  discrete world state (issue #8), e.g.
  `{"garden_gate": {"state": "open", "pressed": false, "depressMm": 1}}`.
  Sparse like body poses — only activities whose flags changed appear, and
  the block is omitted when nothing did — and re-shipped in full on every
  keyframe, so a mid-stream joiner is complete within one keyframe interval
  exactly as it is for poses.
- The header gains **`"activities": [names]`**.

Why it is not merely a convenience: an activity's visible effect usually
lives on a **static body**. The reference gate is a mocap body that ships
once in the scene description and never again, and its pose is not in the
pose stream at all — so for a change like that the flag is the *only*
record anywhere in the stream. Flag semantics are the activity's own; the
website keys its visuals (a swing, a glow) off them and simulates none of it.

### 0.1.0 → 0.2.0 (keyframes recur, and say so)

Additive; a 0.1.0 consumer reading a 0.2.0 stream needs no changes.

- Frames that re-ship **every** dynamic body now carry `"key": true`.
  Frames without it are sparse, exactly as before.
- Those keyframes **recur**, every `keyframeS` sim-seconds (new header
  field, 5.0), rather than occurring only at `t = 0` and after a live
  reconnect.

Why: a browser joining through the website's relay hub is invisible to
the sim — no reconnect fires, so nothing re-keys for it, and every body
that had settled before it arrived is missing from its world forever.
Recurring keyframes bound that wait, and the marker means the hub can
recognize a cache boundary with a field read instead of a set comparison
against the body census.

## Files

| File | What | Regenerate with |
|---|---|---|
| `scene.room_hub.json` | Static scene description of `models/room_hub.xml` | `uv run python -m pluggybot.telemetry.scene` |
| `scene.home_world.json` | The generated home world, with visual hints + zones + spawns (issue #6) | `uv run python -m pluggybot.telemetry.scene models/home_world.xml` |
| `home_world.meta.json` | The generator sidecar the scene JSON was built from | `uv run python -m pluggybot.home.world` |
| `textures/*.png` | The AprilTag textures, decoded from the compiled model | (same command) |
| `telemetry.hub_lifecycle.jsonl.gz` | Full battery-driven mission in **room_hub** (explore → charge → fetch tool → stow) | `MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --record protocol/telemetry.hub_lifecycle.jsonl.gz` |
| `telemetry.home_lifecycle.jsonl.gz` | The same loop in the **home world** (issue #9) with a real **drawing errand** (issue #12) — what the live site serves, and the fixture the canvas-painting code is built against | `MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --world home --errand draw --record protocol/telemetry.home_lifecycle.jsonl.gz` |

**One recording per scene, and they are not interchangeable.** A replayer
picks its scene off the header's `model` field, so playing the room_hub
recording against the home scene poses the robot inside the wrong house —
which renders as a robot driving through walls, not as an error.
`tests/test_telemetry.py` checks each recording's `model` label and each
scene against its committed world.

## Scene description (fetched once)

Transpiled from the **compiled** `MjModel`, so includes/defaults/generated
files are already resolved. Top level:

```jsonc
{
  "protocolVersion": "0.3.0",
  "model": "room_hub",
  "upAxis": "z",              // see "conventions" below
  "bodies": [ ... ],
  "textures": [{"name": "tagtex0", "file": "tagtex0.png", "width": 240, "height": 240}]
}
```

Per body: `name`, `parent` (body name, `null` for the world root),
`dynamic` (whether telemetry will stream poses for it), `robot` (the owning
robot's name, `null` for shared/world bodies), `visual` (the parametric
visual hint — see below), `pos` +
`quat` (**world-frame rest pose**, so a client renders with no
kinematic-tree math: static bodies keep this pose forever, dynamic bodies
get theirs overwritten by telemetry), and `geoms`.

`visual` comes from the world generator's sidecar
(`models/<world>.meta.json`, issue #6): the website renders a parametric
component per hint and falls back to raw primitives for `null` or for a
hint it does not know. Vocabulary v1 (`telemetry.protocol.VISUAL_HINTS`):
`wall`, `fence`, `floor`, `ground`, `whiteboard`, `rack`, `plant`. Adding
a hint is additive; renaming one is a two-repo breaking change. Hints ride
in the sidecar and **never** in geom colors — the robot's cameras render
rgba, so colour-as-encoding would couple perception to art direction.

A generated world may also carry three optional top-level fields, likewise
additive: `zones` (named rectangles, `{name, kind, min:[x,y], max:[x,y]}`),
`spawns` (`name → [x, y, yaw_rad]`), and `boards` — the drawing surfaces, by
the name telemetry uses:

```jsonc
"boards": {"whiteboard_a": {"geom": "board",          // the geom in `bodies`
                            "pos": [-1.97, 1.0, 0.3], // world frame, m
                            "half": [0.01, 0.16, 0.13],  // depth,width,height
                            "heading": 3.14159}}      // outward normal, rad
```

This is what places a `draw` event's polyline: the event names the *board*
(`whiteboard_a`), the scene says which geom that is and how big its face is,
so a canvas is `2 × half[1]` by `2 × half[2]` metres with the polyline's
origin at its centre.

Per geom: `name`, `type` (`box | cylinder | capsule | sphere | plane`),
`size`, `pos` + `quat` (body-local, constant), `rgba`, `texture` (name into
the textures table, or `null`). Geoms with rgba alpha 0 (invisible
collision layers, e.g. the schuko socket wells) are omitted.

### Conventions — conversions already applied

- **Units** meters; quaternions `[w, x, y, z]` (MuJoCo order).
- **Sizes are FULL extents**, converted from MuJoCo's half-extents:
  box `[x, y, z]`; cylinder/capsule `[radius, length]` (capsule length is
  the cylindrical part); sphere `[radius]`; plane `[x, y]` with `0` meaning
  infinite.
- **Cylinder/capsule axis**: MuJoCo's runs along local +Z, ThreeJS's along
  local +Y; each such geom's `quat` already contains the +90° X rotation
  that maps a Y-axis primitive onto the MuJoCo geometry. Planes agree on
  +Z normals and need no fix.
- **The world frame is NOT converted** (`upAxis: "z"`): all poses, here and
  in telemetry, are MuJoCo Z-up world frame. Apply one Z-up → Y-up rotation
  at the ThreeJS scene root.

## Telemetry (JSONL, one object per line)

Line 1 is a header; every later line is a frame at ~`hz` (20) of **sim
time**. A `.gz` suffix means gzip (`zcat` to inspect).

```jsonc
// header
{"type": "header", "protocolVersion": "0.4.0", "model": "room_hub", "hz": 20.0,
 "keyframeS": 5.0,                                      // sim s between keyframes
 "robots": {"pluggybot": ["pluggybot", "head", ...]},   // dynamic bodies per robot
 "world": ["rack", "module_lcd", ...],                  // shared dynamic bodies
 "activities": ["garden_gate"],                         // task state machines
 "boards": ["whiteboard_a", "whiteboard_b"]}            // drawing surfaces

// frame
{"t": 123.45,                                  // sim seconds
 "key": true,                                  // present only on keyframes
 "robots": {"pluggybot": {
   "bodies": {"pluggybot": [x, y, z, qw, qx, qy, qz], ...},   // world-frame
   "state": "EXPLORE",                         // lifecycle state machine
   "status": "EXPLORE -> GO_CHARGE (battery low)",            // the _say line
   "battery": {"frac": 0.61, "watts": 14.2, "charging": false}}},
 "world": {"module_lcd": [x, y, z, qw, qx, qy, qz]},
 "activities": {"garden_gate": {"state": "open", "pressed": false}},
 "boards": {"whiteboard_a": {"programs": ["house"], "strokes": 7,
                             "inkM": 0.459,          // metres of ink laid down
                             "fill": 0.191,          // of the pen's REACH
                             "clears": 1,
                             "clearedAt": "2026-08-16T08:01:09+00:00",
                             "drawnAt": "2026-08-16T08:01:47+00:00"}}}
```

**Frames are sparse.** The first frame is a keyframe carrying every dynamic
body; later frames carry only bodies that moved > 0.5 mm (or the quat
equivalent) since they were last emitted. A body absent from a frame is
unchanged — a replayer holds the last value it saw. `bodies`, `world` and
`activities` are omitted entirely when empty; `state`/`status`/`battery`
ride in every frame. **Activity flags and board state are sparse on the same
rule as poses** and re-ship on every keyframe (0.3.0, 0.4.0). Positions are
rounded to 0.1 mm.

**Keyframes recur** every `keyframeS` sim-seconds (5.0) and are marked
`"key": true`. A consumer that starts reading anywhere in the stream is
complete within one such interval; one that starts at the top can ignore
the marker entirely, since a keyframe is just a frame that happens to
mention everything. At 20 Hz they are 1 frame in 100.

The producer seam: `TelemetryRecorder` (`src/pluggybot/telemetry/recorder.py`)
is a callback on `HubMission.step_hooks` — the same per-physics-step seam
the battery drains through. It decimates 500 Hz of steps to `hz` of frames
and hands them to a writer thread; no serialization or file I/O ever runs
inside a physics step.

## The live stream (webserver v1)

`scripts/serve.py` publishes the same objects over an outbound WebSocket
(`WsPublisher`, `src/pluggybot/telemetry/publisher.py`) — the recorder and
the publisher build frames with the same `FrameBuilder` code (each owns an
instance), so a live consumer and a replayed recording see identical data,
provided both are configured alike: `serve.py --record` passes its
`--keyframe-s` to both, which is the only setting that can make them
disagree. Live-stream rules:

- **Dispatch on `type`; no `type` means frame.** The header and the extra
  message types below carry a `type` field; telemetry frames never do.
  **Consumers must ignore message types they do not recognize** — new
  low-frequency types are additive and do not bump `protocolVersion`
  (only a change to the *shape* of an existing artifact does).
- **Every connection opens with the header, then a keyframe.** Sparse
  frames are deltas against what was previously sent, so whenever
  continuity breaks — a (re)connect, or frames dropped because nobody was
  draining the socket — the next frame re-ships every dynamic body, marked
  `"key": true`. A consumer joining mid-mission starts from nothing and is
  complete within two frame intervals: the re-key is requested by the
  sender thread and honoured by the next physics step, so one sparse frame
  can slip out between the header and the keyframe. Applying it early is
  harmless — the keyframe overwrites everything it touched.
- **A relay hub only needs to cache the last keyframe and the frames
  since it.** The sim connects outbound to the hub and stays connected;
  browsers come and go behind it, invisibly. Nothing re-keys on their
  behalf, which is what recurring keyframes are for. On a browser join,
  replay the cached header, then the cached keyframe, then the frames
  after it, then go live — the cache is bounded by `keyframeS` × `hz`
  (≈100 frames). Also worth caching: the most recent `grid` message per
  robot, so a joiner's map is not blank for a second.
- Frames can drop under load; recordings are the lossless artifact.

**The ingest connection is authenticated.** The publisher presents the
shared secret as an `Authorization: Bearer <token>` request header at the
WebSocket handshake (`scripts/serve.py --token`, or `$PLUGGYWORLD_TOKEN`);
a server that dislikes it should refuse the handshake with `401`. A token
in the URL query works too and needs no producer support — it is just part
of `--endpoint` — but it lands in access logs, so the header is the
default. Refusal is not fatal to the sim: it retries every second like any
other unreachable endpoint, and reports the last failure at exit. Note that
the header only beats the query param on *log* exposure — over plain `ws://`
the bearer crosses the network in cleartext either way, so anything leaving
the host wants `wss://`. An empty token is rejected at startup rather than
sent as no header at all, because a blank `PLUGGYWORLD_TOKEN` would
otherwise publish unauthenticated and read as a wrong secret.

The two live-only message types:

(`draw` and `board_cleared` ride this socket too, and are the two types that
also appear in recordings — see 0.4.0 above.)

```jsonc
// occupancy-grid belief, ~1 Hz per robot: base64 PNG, uint8 cells
// (0 = wall, 255 = free, 127 = unknown), row 0 = y_min edge
{"type": "grid", "t": 123.4, "robot": "pluggybot",
 "extent": [-3, -3, 7, 7],        // [x_min, y_min, x_max, y_max], world m
 "resolution": 0.05, "png": "iVBORw0..."}

// lifecycle narration (the _say lines), as they happen
{"type": "event", "t": 123.4, "robot": "pluggybot",
 "line": "EXPLORE -> GO_CHARGE (battery low)"}
```
