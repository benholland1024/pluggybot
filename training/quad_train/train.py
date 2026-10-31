"""`python -m quad_train.train Pluggy-Humanoid-Walk --env.scene.num-envs 4096 ...`:
registers the humanoid's tasks (`humanoid_train`), then hands off to mjlab's
own training CLI (the same flags, `--agent.resume` included)."""

import humanoid_train  # noqa: F401  (registers the humanoid's tasks)
from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
