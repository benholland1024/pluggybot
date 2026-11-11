"""A get-up checkpoint (rsl_rl's model_N.pt) as the served npz, its metadata off
an npz `quad_train.export` wrote for the same task (the joints it sees, their
defaults, the action's scale, the observation's terms, the gains): mjlab writes
the ONNX of the last save alone, and a checkpoint before it has none.
  python -m quad_train.from_checkpoint MODEL.pt TEMPLATE.npz OUT.npz ITERATION
The normaliser is (x - mean) / (std + 1e-2), rsl_rl's: the npz is identical
to `quad_train.export`'s of the same checkpoint's ONNX (getupwall8's model_9000,
every array to 0.0, #679)."""
import hashlib
import json
import sys

import numpy as np
import torch

ckpt, template, out, iteration = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
state = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
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
  training.update({"iterations": iteration, "env_steps": training["envs"] * iteration * 24,
                   "onnx_sha256": None, "checkpoint": ckpt.split("/")[-2] + "/" + ckpt.split("/")[-1],
                   "checkpoint_sha256": hashlib.sha256(fh.read()).hexdigest()})
arrays["training"] = np.array(json.dumps(training))
assert arrays["obs_mean"].shape == t["obs_mean"].shape
np.savez(out, **arrays)
print("wrote", out, "from", ckpt)
