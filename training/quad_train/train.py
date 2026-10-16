"""`python -m quad_train.train Pluggy-Quad-Flat --env.scene.num-envs 2048 ...`:
registers our tasks (the quadruped's and the humanoid's), then hands off to
mjlab's own training CLI (the same flags, `--agent.resume` included)."""

import humanoid_train  # noqa: F401  (registers the humanoid's tasks)
import quad_train  # noqa: F401  (registers the quadruped's tasks)
from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
