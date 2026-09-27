"""ONNX -> npz: the trained actor as plain arrays, for `pluggybot.legs.policy`.

  python -m quad_train.export RUN.onnx ../models/quadruped_policy.npz \\
      --gpu "GTX 1660 Super" --wall "1 h 20 min" --envs 2048 --iterations 2000

mjlab writes an ONNX file at every checkpoint: a normaliser (Sub, Div) and an
MLP of Gemm + Elu. Its initializers are read with `onnx` alone (protobuf, no
torch), together with the metadata the file carries (joint order, default
pose, action scale, observation terms), and the training facts #377 asks to
be kept with every policy (the GPU, the wall time, the environment count).
"""

import argparse
import hashlib
import json

import numpy as np
import onnx
from onnx import numpy_helper

from quad_train import robot


def load(path: str) -> dict:
  model = onnx.load(path)
  init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
  meta = {p.key: p.value for p in model.metadata_props}
  nodes = list(model.graph.node)
  ops = [n.op_type for n in nodes]
  assert ops[:2] == ["Sub", "Div"], ops
  out = {"obs_mean": init[nodes[0].input[1]].astype(np.float32).reshape(-1),
         "obs_div": init[nodes[1].input[1]].astype(np.float32).reshape(-1)}
  layers = [n for n in nodes if n.op_type == "Gemm"]
  for k, gemm in enumerate(layers):
    w = init[gemm.input[1]].astype(np.float32)
    trans_b = next((a.i for a in gemm.attribute if a.name == "transB"), 0)
    out[f"w{k}"] = w.T if trans_b else w      # always (in, out)
    out[f"b{k}"] = init[gemm.input[2]].astype(np.float32)
  acts = {n.op_type for n in nodes[2:] if n.op_type != "Gemm"}
  assert acts == {"Elu"}, acts
  out["layers"] = np.array(len(layers))
  out["meta"] = np.array(json.dumps(meta))
  return out


def forward(arrays: dict, obs: np.ndarray) -> np.ndarray:
  """The numpy forward pass, as `pluggybot.legs.policy.WalkingPolicy.act`."""
  x = (obs.astype(np.float64) - arrays["obs_mean"]) / arrays["obs_div"]
  n = int(arrays["layers"])
  for k in range(n):
    x = x @ arrays[f"w{k}"].astype(np.float64) + arrays[f"b{k}"]
    if k < n - 1:
      x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
  return x


def check(path: str, arrays: dict, samples: int = 256) -> float:
  """Max |numpy - ONNX| over random observations, by ONNX's own reference
  evaluator: the export refuses to write weights the file does not run."""
  from onnx.reference import ReferenceEvaluator
  ref = ReferenceEvaluator(path)
  rng = np.random.default_rng(0)
  obs = rng.normal(size=(samples, arrays["obs_mean"].shape[0])).astype(np.float32)
  worst = 0.0
  for row in obs:
    want = ref.run(None, {"obs": row[None]})[0][0]
    worst = max(worst, float(np.abs(forward(arrays, row) - want).max()))
  return worst


def main(argv=None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx")
  ap.add_argument("out")
  ap.add_argument("--gpu", required=True)
  ap.add_argument("--wall", required=True, help="training wall time")
  ap.add_argument("--envs", type=int, required=True)
  ap.add_argument("--iterations", type=int, required=True)
  ap.add_argument("--task", default="Pluggy-Quad-Flat")
  args = ap.parse_args(argv)
  arrays = load(args.onnx)
  err = check(args.onnx, arrays)
  assert err < 1e-4, f"numpy and ONNX disagree by {err}"
  print(f"numpy matches the ONNX file to {err:.1e}")
  with open(args.onnx, "rb") as fh:
    onnx_sha = hashlib.sha256(fh.read()).hexdigest()
  arrays["training"] = np.array(json.dumps({
    "task": args.task, "gpu": args.gpu, "wall": args.wall,
    "envs": args.envs, "iterations": args.iterations,
    "env_steps": args.envs * args.iterations * 24, "onnx_sha256": onnx_sha,
    "train_dt": robot.TRAIN_DT, "decimation": 4,
    "stiffness": robot.STIFFNESS, "damping": robot.DAMPING,
  }))
  np.savez(args.out, **arrays)
  print(f"wrote {args.out}: {int(arrays['layers'])} layers, "
        f"obs {arrays['obs_mean'].shape[0]}, actions {arrays[f'b{int(arrays['layers']) - 1}'].shape[0]}")


if __name__ == "__main__":
  main()
