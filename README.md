# pluggybot

Start one of the various scripts:
```bash
uv run python scripts/teleop.py      # Teleop test the robot
uv run python scripts/map_teleop.py  # Teleop the robot while updating /map.png
uv run python scripts/explore.py     # Run the frontier exploration script (also updates map)

uv run python scripts/lifecycle.py    # THE loop the project is named after:
                                      # explore -> find an outlet -> dock ->
                                      # charge -> resume. Saves map.png/views.png

uv run python scripts/hub_mission.py --view  # Explore, find + use hub
uv run python scripts/hub_lifecycle.py --view  # Hub-era battery loop: explore,
                                      # fetch a tool, use it, stow it, charge

uv run python scripts/draw.py --view --fast
uv run python scripts/draw.py --shape square

uv run python scripts/pickup.py --view   # claw module: pick a block off the
                                      # floor; saves pickup.png

MUJOCO_GL=egl uv run python scripts/module_power.py  # Tool power across the
                                      # coupling; saves module_power.png
```

Headless (no window) runs want `MUJOCO_GL=egl` in front; `--view` runs want it
left off.

## Recording a demo

`draw.py` and `pickup.py` take `--record PATH` (`.mp4` or `.gif`) and render
720p video straight from the sim — no screen capture. The camera is the
filmstrip's tracking camera, sampled on the **sim** clock, so playback speed
is exact and `--record-speed` buys timelapse or slow motion for free:

```bash
MUJOCO_GL=egl uv run python scripts/draw.py --shape square \
    --record draw.mp4 --record-speed 3        # 66 s of sim -> 22 s of video
MUJOCO_GL=egl uv run python scripts/pickup.py \
    --record pickup.gif --record-speed 4 --record-fps 20
```

`--record` works with or without `--view` (the video comes from its own
offscreen renderer either way) and never changes what the demo reports.
`pickup.py` carries `claw_eye` as a picture-in-picture — the camera on the
tool itself, which no screen recording could reach. GIFs are downscaled to
640 px and encoded through a single shared palette; for Reddit prefer `.mp4`,
since it transcodes uploaded GIFs to video anyway.

View the world:
```bash
uv run python -m mujoco.viewer --mjcf=models/world.xml
```

Run scripts in /tests/ :
```bash
uv run pytest -v
# or, to output print statements:
uv run pytest -vs
```

# Running Pluggyworld on rooftop-media.org

In this project, start it like so:
```bash
PLUGGYWORLD_TOKEN=dev-token-change-me MUJOCO_GL=osmesa \
  uv run python scripts/serve.py --endpoint ws://localhost:3000/api/pluggyworld/ingest
```

Then, in `rooftop-media-2026`, start it with `npm run dev`

# Outlet visual recognition with Yolo CNN

Regenerate the training data:
```bash
MUJOCO_GL=egl uv run python scripts/generate_outlet_dataset.py --count 1200
```

Train the outlet detector with YOLO:
```bash
uv run yolo detect train data=datasets/outlets/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640
```

Test predictions with YOLO:
```bash
uv run yolo predict model=runs/detect/train/weights/best.pt source=datasets/outlets/images/val
```
