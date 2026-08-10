# pluggybot

Start one of the various scripts:
```bash
uv run python scripts/teleop.py      # Teleop test the robot
uv run python scripts/map_teleop.py  # Teleop the robot while updating /map.png
uv run python scripts/explore.py     # Run the frontier exploration script (also updates map)

uv run python scripts/hub_mission.py --view  # Explore, find + use hub

uv run python scripts/draw.py --view --fast
uv run python scripts/draw.py --shape square
```

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
