"""Checkpoints of a get-up run in the TRAINING physics, each from chosen starts
against the box, its mean action: the share standing free at 14 s, one row a
checkpoint. Each case's env is built once.
  python -m humanoid_train.scan RUN_DIR [EVERY] [N]   (from training/)"""
import re
import sys
from pathlib import Path

import torch
import humanoid_train  # noqa: F401
from humanoid_train import getup
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

run = Path(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 200
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
ckpts = sorted(run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
first = int(re.findall(r"\d+", ckpts[0].name)[0])
ckpts = [p for p in ckpts if (int(re.findall(r"\d+", p.name)[0]) - first) % every == 0] + [ckpts[-1]]
ckpts = list(dict.fromkeys(ckpts))
LYING = {"beside": 1.0, "head": 0.0, "feet": 0.0}
CASES = {
  "side front": ({"side": 1.0}, {"share": 1.0, "front": 1.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.05)}}),
  "back, box L": ({"back": 1.0}, {"share": 1.0, "left": 1.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}),
  "back, box R": ({"back": 1.0}, {"share": 1.0, "left": 0.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}),
  "front, box L": ({"front": 1.0}, {"share": 1.0, "left": 1.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}),
  "front, box R": ({"front": 1.0}, {"share": 1.0, "left": 0.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}),
  "side back": ({"side": 1.0}, {"share": 1.0, "front": 0.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.05)}}),
  "open: back": ({"back": 1.0}, {"share": 0.0}),
  "open: side": ({"side": 1.0}, {"share": 0.0}),
  "open: tip": ({"tipping": 1.0}, {"share": 0.0, "tipping_share": 0.0}),
  "tip along": ({"tipping": 1.0}, {"share": 1.0, "tipping_share": 1.0, "tipping": {"into": 0.0, "along": 1.0}}),
}


def actor_of(path):
  state = torch.load(path, map_location="cuda:0", weights_only=False)["actor_state_dict"]

  def actor(obs):
    x = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2)
    for k in (0, 2, 4):
      x = torch.nn.functional.elu(x @ state[f"mlp.{k}.weight"].T + state[f"mlp.{k}.bias"])
    return x @ state["mlp.6.weight"].T + state["mlp.6.bias"]
  return actor


rows = {p.name: {} for p in ckpts}
for name, (starts, box) in CASES.items():
  cfg = load_env_cfg("Pluggy-Humanoid-Getup-Wall", play=True)
  cfg.scene.num_envs = n
  cfg.episode_length_s = 20.0
  draws = {**getup.OBSTACLE, **box}
  for key in ("gap_m", "lying", "tipping"):
    draws[key] = {**getup.OBSTACLE[key], **box.get(key, {})}
  cfg.events["reset_base"].params["starts"] = starts
  cfg.events["reset_base"].params["obstacle"] = draws
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  for p in ckpts:
    actor = actor_of(p)
    obs, _ = env.reset()
    with torch.no_grad():
      for k in range(700):
        obs, *_ = env.step(actor(obs["actor"]))
    rows[p.name][name] = float(getup._standing(env).float().mean())
  env.close()
  print("done", name, flush=True)
print(f"{'checkpoint':16s}" + "".join(f"{c:>13s}" for c in CASES))
for p in ckpts:
  print(f"{p.name:16s}" + "".join(f"{rows[p.name][c]:13.2f}" for c in CASES))
