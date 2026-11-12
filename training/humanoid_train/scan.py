"""Checkpoints of a get-up run in the TRAINING physics, each from chosen starts
against the box, its mean action: the share standing free at 14 s, one row a
checkpoint (a world that blew up or went NaN on the way, and was started
again, counts as down, and the table says how many did). Each case's env is
built once and flies every checkpoint at once, each its own block of worlds
(a BASE checkpoint first, to read the others against in the same pass).
  python -m humanoid_train.scan RUN_DIR [EVERY] [N] [BASE.pt]   (from training/)"""
import json
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
base = Path(sys.argv[4]) if len(sys.argv) > 4 else None
saved = {int(m.group(1)): p for p in run.glob("model_*.pt")
         if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))}
if not saved:
  sys.exit(f"no model_N.pt in {run}")
first, last = min(saved), max(saved)
ckpts = ([base] if base else []) + [saved[i] for i in sorted(saved)
                                    if (i - first) % every == 0 or i == last]
LYING = {"beside": 1.0, "head": 0.0, "feet": 0.0}
POSES = json.loads(getup.POSES.read_text())["poses"]
SITTING = tuple(k for k, p in POSES.items() if "joint_vel" not in p and not p["against"])
LEANING = tuple(k for k, p in POSES.items() if "joint_vel" not in p and p["against"])
FALLING = tuple(k for k, p in POSES.items() if "joint_vel" in p)
#: (starts, the box's draws over `getup.OBSTACLE`, the posed starts' names).
CASES = {
  "side front": ({"side": 1.0}, {"share": 1.0, "front": 1.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.05)}}, None),
  "back, box L": ({"back": 1.0}, {"share": 1.0, "left": 1.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}, None),
  "back, box R": ({"back": 1.0}, {"share": 1.0, "left": 0.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}, None),
  "front, box L": ({"front": 1.0}, {"share": 1.0, "left": 1.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}, None),
  "front, box R": ({"front": 1.0}, {"share": 1.0, "left": 0.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.1)}}, None),
  "side back": ({"side": 1.0}, {"share": 1.0, "front": 0.0, "lying": LYING, "gap_m": {"beside": (0.0, 0.05)}}, None),
  "open: back": ({"back": 1.0}, {"share": 0.0}, None),
  "open: side": ({"side": 1.0}, {"share": 0.0}, None),
  "open: tip": ({"tipping": 1.0}, {"share": 0.0, "tipping_share": 0.0}, None),
  "tip along": ({"tipping": 1.0}, {"share": 1.0, "tipping_share": 1.0, "tipping": {"into": 0.0, "along": 1.0}}, None),
  # #679's second round: where the wall policy sat or leant, and its falls into them
  "sit, open": ({"posed": 1.0}, {"posed_share": 0.0}, SITTING),
  "sit, box": ({"posed": 1.0}, {"posed_share": 1.0}, SITTING),
  "leaning": ({"posed": 1.0}, {}, LEANING),
  "falls": ({"posed": 1.0}, {}, FALLING),
  "tip far": ({"tipping": 1.0}, {"tipping_share": 1.0, "tipping": {"into": 0.0, "along": 0.0, "far": 1.0}}, None),
}


def actor_of(path):
  state = torch.load(path, map_location="cuda:0", weights_only=False)["actor_state_dict"]

  def actor(obs):
    x = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2)
    for k in (0, 2, 4):
      x = torch.nn.functional.elu(x @ state[f"mlp.{k}.weight"].T + state[f"mlp.{k}.bias"])
    return x @ state["mlp.6.weight"].T + state["mlp.6.bias"]
  return actor


actors = [actor_of(p) for p in ckpts]
names = [("BASE " if base and k == 0 else "") + p.name for k, p in enumerate(ckpts)]
rows = {name: {} for name in names}
lost = {name: 0 for name in names}
for case, (starts, box, poses) in CASES.items():
  cfg = load_env_cfg("Pluggy-Humanoid-Getup-Wall2-Mirror", play=True)
  cfg.scene.num_envs = n * len(ckpts)
  cfg.episode_length_s = 20.0
  draws = {**getup.OBSTACLE, **box}
  for key in ("gap_m", "lying", "tipping"):
    draws[key] = {**getup.OBSTACLE[key], **box.get(key, {})}
  cfg.events["reset_base"].params["starts"] = starts
  cfg.events["reset_base"].params["obstacle"] = draws
  cfg.events["reset_base"].params["poses"] = poses
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  obs, _ = env.reset()
  broke = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  with torch.no_grad():
    for _ in range(700):
      seen = obs["actor"]
      act = torch.cat([actor(seen[k * n:(k + 1) * n]) for k, actor in enumerate(actors)])
      obs, _, terminated, _, _ = env.step(act)
      broke |= terminated
  up = getup._standing(env) & ~broke
  for k, name in enumerate(names):
    rows[name][case] = float(up[k * n:(k + 1) * n].float().mean())
    lost[name] += int(broke[k * n:(k + 1) * n].sum())
  env.close()
  print("done", case, flush=True)
print(f"{'checkpoint':21s}" + "".join(f"{c:>13s}" for c in CASES) + f"{'blew up':>9s}")
for name in names:
  print(f"{name:21s}" + "".join(f"{rows[name][c]:13.2f}" for c in CASES) + f"{lost[name]:9d}")
print("JSON " + json.dumps({"cases": list(CASES), "rows": rows, "lost": lost}))
