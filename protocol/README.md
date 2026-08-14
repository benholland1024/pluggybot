# The PluggyWorld wire protocol — canonical fixtures

This directory is the data contract between pluggybot (the producer) and
the `rooftop-media-2026` website (the consumer). The website repo vendors
copies of these files into its test fixtures and never imports pluggybot
code; both repos' tests run against the same recorded artifacts. Design
doc: `rooftop-media-2026/docs/pluggyworld.md`, § "The scene protocol" and
§ "Repo topology"; the website-side spec lives with its protocol issue.

**Versioning.** Every artifact carries `protocolVersion`
(`pluggybot.telemetry.protocol.PROTOCOL_VERSION`, currently `0.1.0`).
Bumping it is a deliberate two-repo event: change the shape, bump the
version, regenerate these fixtures, and re-vendor them in the website repo.
`tests/test_telemetry.py` fails if the committed fixtures drift from the
committed world or the committed version.

## Files

| File | What | Regenerate with |
|---|---|---|
| `scene.room_hub.json` | Static scene description of `models/room_hub.xml` | `uv run python -m pluggybot.telemetry.scene` |
| `textures/*.png` | The AprilTag textures, decoded from the compiled model | (same command) |
| `telemetry.hub_lifecycle.jsonl.gz` | Full battery-driven mission recording (explore → charge → fetch tool → stow) | `MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --record protocol/telemetry.hub_lifecycle.jsonl.gz` |

## Scene description (fetched once)

Transpiled from the **compiled** `MjModel`, so includes/defaults/generated
files are already resolved. Top level:

```jsonc
{
  "protocolVersion": "0.1.0",
  "model": "room_hub",
  "upAxis": "z",              // see "conventions" below
  "bodies": [ ... ],
  "textures": [{"name": "tagtex0", "file": "tagtex0.png", "width": 240, "height": 240}]
}
```

Per body: `name`, `parent` (body name, `null` for the world root),
`dynamic` (whether telemetry will stream poses for it), `robot` (the owning
robot's name, `null` for shared/world bodies), `visual` (the parametric
visual hint slot — `null` until the park generator fills it), `pos` +
`quat` (**world-frame rest pose**, so a client renders with no
kinematic-tree math: static bodies keep this pose forever, dynamic bodies
get theirs overwritten by telemetry), and `geoms`.

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
{"type": "header", "protocolVersion": "0.1.0", "model": "room_hub", "hz": 20.0,
 "robots": {"pluggybot": ["pluggybot", "head", ...]},   // dynamic bodies per robot
 "world": ["rack", "module_lcd", ...]}                  // shared dynamic bodies

// frame
{"t": 123.45,                                  // sim seconds
 "robots": {"pluggybot": {
   "bodies": {"pluggybot": [x, y, z, qw, qx, qy, qz], ...},   // world-frame
   "state": "EXPLORE",                         // lifecycle state machine
   "status": "EXPLORE -> GO_CHARGE (battery low)",            // the _say line
   "battery": {"frac": 0.61, "watts": 14.2, "charging": false}}},
 "world": {"module_lcd": [x, y, z, qw, qx, qy, qz]}}
```

**Frames are sparse.** The first frame is a keyframe carrying every dynamic
body; later frames carry only bodies that moved > 0.5 mm (or the quat
equivalent) since they were last emitted. A body absent from a frame is
unchanged — a replayer holds the last value it saw. `bodies` and `world`
are omitted entirely when empty; `state`/`status`/`battery` ride in every
frame. Positions are rounded to 0.1 mm.

The producer seam: `TelemetryRecorder` (`src/pluggybot/telemetry/recorder.py`)
is a callback on `HubMission.step_hooks` — the same per-physics-step seam
the battery drains through. It decimates 500 Hz of steps to `hz` of frames
and hands them to a writer thread; no serialization or file I/O ever runs
inside a physics step.
