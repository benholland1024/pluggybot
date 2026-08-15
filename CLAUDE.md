# PluggyBot — notes for Claude

Simulated self-charging robot in MuJoCo. Before doing anything, read:
- `docs/PluggyPlan.md` — goals, milestone status, architecture
- `docs/SimNotes.md` — hard-won simulation lessons; read BEFORE touching `models/` or contact/actuator params
- `docs/Parts.md` — locked hardware decisions and the sim parameters they feed

## Working style

- Explain in prose, at the level of "a teammate catching up": name the
  concepts (running average, pinhole projection, convex decomposition) rather
  than assuming them, and say what a number means, not just what it is.
- Ben may still claim ML training runs, but Claude runs them by default now.
- Verify physics claims empirically (headless probes, filmstrip renders via
  offscreen Renderer) rather than by reasoning alone; it has won every time.
  When a result looks good, try to break it before believing it — the MSAA
  label bug, the stale-dataset contamination, and the spin-collision gap were
  all found this way, and two of them were hiding behind green metrics.
- Every debugged failure becomes a pytest assertion, and the assertion must be
  shown to fail without the fix — a regression test that cannot fail is décor.

## Commands

- Tests: `MUJOCO_GL=egl uv run pytest -q`; lint: `uv run ruff check src/ scripts/ tests/`
- Demos: `scripts/teleop.py`, `scripts/map_teleop.py`, `scripts/explore.py [--headless]`
  (milestone-4 mapping demo — kept as the minimal repro; `lifecycle.py` is the
  full mission), `scripts/spot_outlets.py` (detector → landmarks),
  `scripts/lifecycle.py [--headless] [--battery-wh W]` (full mission loop:
  explore → dock → charge → resume; battery-driven since milestone 7 —
  `--explore-budget N` remains as a timer override, and DOCK is real
  physics — plug seats in the socket), `scripts/schuko_spike.py`
  (docking tolerance sweep), `scripts/hub_spike.py` (milestone-8 tool-coupling
  tolerance sweep; `--film` for a filmstrip), `scripts/noslip_spike.py`
  (issue-3 solver-policy sweep: coupling/schuko seat, jittered robot swap,
  grip creep, pen square, step cost under candidate `noslip_iterations`
  values; `--no-brake` reproduces the before-fix rows), `scripts/hub_swap.py` (robot
  swaps a module at the hub in `models/hub_world.xml`),
  `scripts/draw.py` (the drawing tool: fetch the pen module from bay C, carry
  it to a board, plot a figure; saves `draw.png` — filmstrip + commanded-vs-
  traced overlay + error stats. `--view` watches it live and skips the
  filmstrip, `--shape square` for the more diagnostic figure),
  `scripts/pickup.py` (the claw module: fetch it from bay D, grip a block off
  the floor, carry it, set it down; saves `pickup.png`. `--view` watches live.
  Full pick-carry-place verified),
  `scripts/module_power.py` (module electrical interface: runs the errand and
  saves `module_power.png` — filmstrip with the module painted live/dead plus
  a per-pole continuity timeline; `--bare` for the faster hub_world version),
  `scripts/hub_mission.py`
  (the milestone-8 story: navigate room_hub → fine-align → swap → return;
  `--view` opens the MuJoCo viewer and paces it to real time, `--fast` skips
  the pacing), `scripts/hub_lifecycle.py` (the hub-era battery-driven loop:
  explore → fetch a tool → use it → stow it → charge at the hub; `--view`,
  `--battery-wh W`, `--record out.jsonl.gz` writes a PluggyWorld telemetry
  recording — issue #4), `scripts/serve.py --endpoint ws://host:port`
  (webserver v1, issue #5: the hub lifecycle headless, paced to real time,
  streaming protocol frames + grid PNGs + event lines over an outbound
  WebSocket — the sim never blocks on the socket; `--rate X`, `--free-run`
  measures the machine's real-time multiple; `--token` / `$PLUGGYWORLD_TOKEN`
  is the website's ingest secret; docs/Webserver.md),
  `scripts/ws_sink.py` (dummy sink for serve.py: message counts + received
  frame-gap stats + keyframe spacing; `--token` makes it refuse an
  unauthenticated publisher, like the real ingest path)
- PluggyWorld protocol fixtures (`protocol/`, issue #4) are GENERATED — the
  scene JSON + tag textures via `uv run python -m pluggybot.telemetry.scene`
  (rerun after changing any room_hub geometry — the fixture test fails when
  stale), the telemetry recording via `MUJOCO_GL=egl uv run python
  scripts/hub_lifecycle.py --record protocol/telemetry.hub_lifecycle.jsonl.gz`.
  Format + versioning rules in `protocol/README.md`; a `protocolVersion` bump
  is a deliberate two-repo event (the website repo vendors these fixtures).
- The HOME world is GENERATED (issue #6) — regenerate `models/home_world.xml`
  + `models/home_world.meta.json` with `uv run python -m pluggybot.home.world`
  after changing any layout constant in `src/pluggybot/home/world.py` (the
  committed pair is tested against the generator, so a stale file fails).
  Layout, visual hints, zones, spawns, board specs and the battery re-tune
  all come from that ONE module. Run it: `--world home` on
  `scripts/hub_lifecycle.py` (explore → errand → charge in the house), and
  `scripts/home_draw.py` (fetch the pen → draw on a wall-mounted whiteboard
  → try to stow; `--board whiteboard_b`, `--shape circle`, `--view`).
  ⚠ The pen's STOW after a navigated errand is a known pre-existing failure
  (it fails in room_hub too) — see SimNotes "The home world … and a stow gap".
- **Visual hints are a two-repo contract.** `telemetry.protocol.VISUAL_HINTS`
  is the vocabulary; the sidecar's `visualHints` may only use those strings,
  and `scene_dict` raises on anything else. Adding a hint is additive (the
  website falls back to raw primitives); renaming one is a breaking change in
  both repos. NEVER encode hints as geom colors — the robot's cameras render
  rgba, and colour-as-encoding couples the tag detector to the website's art.
- Hub worlds are GENERATED — regenerate `models/hub_world.xml` +
  `models/hub_rack.xml` with `uv run python -m pluggybot.hub.coupling` after
  changing any rack geometry. `models/room_1_scenery.xml` is the shared floor
  plan behind both `room_1.xml` (plug robot) and `room_hub.xml` (fork robot);
  edit scenery there, never in one room only.
- `--views` on teleop.py / map_teleop.py / lifecycle.py saves `views.png`
  (stereo pair + map + dock camera, issue #1) alongside `map.png`; ~15 ms/save
- **Demo video**: `--record PATH` (`.mp4`/`.gif`) on `scripts/draw.py` and
  `scripts/pickup.py` renders 720p footage offscreen via `viz.Recorder`, with
  `--record-fps` and `--record-speed` (sim seconds per played second).
  Frames are STREAMED to the encoder, never buffered — a 90 s demo at 30 fps
  is ~2700 frames, which is ~7 GB of 720p RGB if you hold them. Two rules the
  tests guard (`tests/test_viz.py`): recording must never step the sim (an
  end-of-clip "hold on the final pose" belongs AFTER the result dict, or it
  silently shifts the reported settle state), and the render size must sit on
  the 16-px macroblock grid or ffmpeg resamples the frames behind you.
  ⚠ The recorder's camera is NOT always the filmstrip's: for `draw.py` the
  filmstrip's az=150 sits BEHIND the board (a thin slab at x=1.30, drawn on
  its -x face), so the whole drawing phase renders as a grey rectangle. The
  video pans to az=60 during the drive. Pick angles by sweeping azimuth at the
  moment of contact, not by reasoning about the geometry.
- RL docking (milestone 6): train
  `MUJOCO_GL=egl uv run python scripts/train_docking.py` (SAC over
  `pluggybot.envs.DockEnv`; checkpoints under `runs/docking/`), score
  `MUJOCO_GL=egl uv run python scripts/eval_docking.py --trials 24`
  (scripted DOCK vs RL policy on identical room_1 trials with the real YOLO)
- Sockets: `models/schuko_sockets.xml` is GENERATED (invisible collision
  layer for room_1's outlets) — regenerate with
  `uv run python -m pluggybot.docking.schuko` after moving an outlet
- Dataset (deterministic; `datasets/` is gitignored, and the generator wipes it
  first — regenerating into a dirty dir once contaminated 195 labels):
  `MUJOCO_GL=egl uv run python scripts/generate_outlet_dataset.py --count 1200`
- Eval (the one that matters — the val split shares the training generator and
  scored 0.99 mAP while calling a light switch an outlet):
  `MUJOCO_GL=egl uv run python scripts/eval_detector.py --poses 1000`
- Train: `uv run yolo detect train data=datasets/outlets/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640`
  (torch is pinned to the cu128 index in `pyproject.toml`: the driver here is
  CUDA 12.8, and PyPI's default cu130 build silently falls back to CPU)

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- `models/world.xml` = bare world for physics tests; `playground.xml` /
  `room_1.xml` add scenery for humans and mapping. Never put scenery in the
  test world.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy, ix]`.
- Odometry tracks the axle midpoint; `qpos` tracks the body origin 8 cm ahead.
- **Position setpoints are always RAMPED, never written across a gap.** A stiff
  servo handed a step delivers it as an impulse: it has thrown a module off the
  fork and batted a gripped block out of the jaws. `control.slew` does this for
  wheels; `ClawTool.set_lift`/`jaws` and `PenPlotter.ramp` for the rest.
- **One solver policy: `noslip_iterations` is 0, always and everywhere**
  (issue #3; sweep table in SimNotes). Never phase-toggle solver modes:
  always-on noslip ≥ 1 half-seats the jittered coupling (on the fork but not
  electrically powered), and a runtime toggle is global state that leaks
  across fixtures — and, in the shared PluggyWorld model, across robots.
  Creep under sustained load is fixed at its actual source, per-part:
  `coupling.GRIP_SOLIMP` where a **contact** drifts (the claw's jaw pads,
  slip −21.7 mm → −0.13 mm), and wheel-joint `frictionloss` where a
  **joint** rolls — a velocity servo commanded 0 resists speed, not force,
  so without the gearbox's parking brake the parked base walks under tool
  loads (the plotter's square: 63 % → 99 % inked, no solver pass, no cost).
  ⚠ The swap's travel constants implicitly contain mm-scale wheel slip;
  anything touching wheel contact or joint friction must re-verify the
  bay-C pick and the mission stow. ⚠ The brake also creates a stiction
  DEADBAND: wheel-speed commands under `frictionloss/kv` (0.1 rad/s) move a
  stopped wheel not at all, so P-turn controllers must go through
  `control.turn_command` (breakaway floor) — a raw `gain × err` command
  crawls (55 s to settle 0.23°, measured). `PenPlotter.contact_physics` /
  `ClawTool.grasp_physics` are deprecated no-ops;
  `tests/test_noslip_policy.py` guards all of it.
- Contact params combine as the elementwise **MAX** unless `priority` is set —
  so a low `friction` without `priority="1"` does nothing. This has bitten the
  caster, the pen pads and the coupling peg.
