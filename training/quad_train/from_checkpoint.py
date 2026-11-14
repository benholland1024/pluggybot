"""A get-up checkpoint (rsl_rl's model_N.pt) as the served npz, its metadata off
an npz `quad_train.export` wrote for the same task (the joints it sees, their
defaults, the action's scale, the observation's terms, the gains): mjlab writes
the ONNX of the last save alone, and a checkpoint before it has none.
  python -m quad_train.from_checkpoint MODEL.pt TEMPLATE.npz OUT.npz [TASK [WALL]]
(TASK and WALL: the task it trained on and its wall-clock line, where they are not the
template's).
The normaliser is (x - mean) / (std + 1e-2), rsl_rl's: the npz is identical
to `quad_train.export`'s of the same checkpoint's ONNX (getupwall8's model_9000,
every array to 0.0, #679)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ckpt, template, out = sys.argv[1], sys.argv[2], sys.argv[3]
task, wall = (sys.argv[4:] + [None, None])[:2]
saved = torch.load(ckpt, map_location="cpu", weights_only=False)
state, iteration = saved["actor_state_dict"], int(saved["iter"])
t = np.load(template)
arrays = {"obs_mean": state["obs_normalizer._mean"].numpy().ravel().astype(np.float32),
          "obs_div": (state["obs_normalizer._std"].numpy().ravel() + 1e-2).astype(np.float32)}
for k, layer in enumerate((0, 2, 4, 6)):
  arrays[f"w{k}"] = state[f"mlp.{layer}.weight"].numpy().T.astype(np.float32)
  arrays[f"b{k}"] = state[f"mlp.{layer}.bias"].numpy().astype(np.float32)
arrays["layers"] = np.array(4)
arrays["meta"] = t["meta"]
training = json.loads(str(t["training"]))
with open(ckpt, "rb") as fh:
  # iterations and steps the lineage's, as `export` counts them (the resumed runs' too)
  training.update({"iterations": iteration, "env_steps": training["envs"] * iteration * 24,
                   "onnx_sha256": None,
                   "checkpoint": f"{Path(ckpt).resolve().parent.name}/{Path(ckpt).name}",
                   "checkpoint_sha256": hashlib.sha256(fh.read()).hexdigest()})
if task:
  training["task"] = task
if wall:
  training["wall"] = wall
arrays["training"] = np.array(json.dumps(training))
# ...the template's shapes, every layer, and so its inputs, joints and actions
for key in ("obs_mean", "obs_div", *(f"{w}{k}" for k in range(4) for w in "wb")):
  assert arrays[key].shape == t[key].shape, (key, arrays[key].shape, t[key].shape)
assert len(training["action_joints"]) == arrays["b3"].shape[0]
out = out if out.endswith(".npz") else out + ".npz"   # np.savez's own rule
np.savez(out, **arrays)
print("wrote", out, "from", ckpt)
