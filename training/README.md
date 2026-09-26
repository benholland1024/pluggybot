# Training the quadruped's walking policy (#377)

A uv project of its own. Nothing here is imported by `pluggybot`, and none
of it enters the serving image: the served sim runs the exported policy in
plain numpy (`pluggybot.legs.policy`).

- **mjlab 1.5.x** (MuJoCo Warp + PyTorch + rsl_rl), because it is the line
  that pins `mujoco ~=3.10.0`, the served sim's own version, and because the
  actuator honesty #377 needs is configuration there: a DC-motor torque-speed
  curve, command delay, randomised armature, friction and effort limits.
  Playground was ~2.5× faster on this card but carries none of that stock
  (the trainer comparison is in the PR for #377).
- It reads the body from `models/quadruped.xml` and `models/quadruped.json`,
  which `uv run python -m pluggybot.legs.model` writes: mjlab 1.5 caps numpy
  below pluggybot's, so the two cannot share an environment.

```bash
cd training && uv sync
# train (checkpoints every 50 iterations; --agent.resume True --agent.load-run
# <dir> carries on after an interruption)
WARP_CACHE_PATH=... .venv/bin/python -m quad_train.train Pluggy-Quad-Flat \
  --env.scene.num-envs 2048 --agent.max-iterations 2000 \
  --agent.logger tensorboard --agent.run-name flat --log-root <runs>
# export: numpy is checked against the ONNX file before anything is written
.venv/bin/python -m quad_train.export <run>/<run>.onnx ../models/quadruped_policy.npz \
  --gpu "<card>" --wall "<h:mm>" --envs 2048 --iterations 2000
# fly it in the served sim's physics, and hash it twice
cd .. && uv run python scripts/quad_spike.py --policy
uv run python scripts/quad_spike.py --determinism
```

**Tasks:** `Pluggy-Quad-Flat` (walking on flat ground) and
`Pluggy-Quad-Rough` (stairs up and down to 0.20 m risers on a 0.28 m tread,
blocks to 0.15 m, rough ground, slopes; blind — the critic sees the terrain,
the actor does not).

⚠ **The GPU is shared** with the test suite, whose EGL renderers hold 3–4 GB
of a 6 GB card: stay at ≤ 2048 environments while it runs, or take
`flock -x ~/pluggybot-worktrees/.box.lock` for a full run.
