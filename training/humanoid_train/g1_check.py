"""Rule 6's check (#507): mjlab's stock G1 velocity policy, trained once,
flown OUTSIDE mjlab -- its ONNX file read into plain arrays and run in numpy
(`quad_train.export`, the path every policy of ours takes to the served
sim), stepping MuJoCo's C engine, not MuJoCo Warp -- at the step it trained
on and at the served sim's 2 ms. A body its maker walks, through our whole
pipeline: if it walks here, a later failure is our design's or our
training's, never the pipeline's.

  .venv/bin/python -m humanoid_train.g1_check RUN/policy.onnx [--dt 0.002]

The scene is mjlab's own (`Scene`, the task's play configuration, compiled
on the CPU), so the G1 and its actuators are exactly what it trained on.
"""

import argparse
import json

import mujoco
import numpy as np

from quad_train import export

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
#: The legs it is asked to walk: (seconds, forward m/s, sideways, turn rad/s).
SEGMENTS = ((3.0, 0.0, 0.0, 0.0), (6.0, 0.5, 0.0, 0.0), (6.0, 1.0, 0.0, 0.0),
            (5.0, 0.0, 0.0, 0.4), (5.0, -0.5, 0.0, 0.0), (5.0, 0.0, 0.4, 0.0))


def build(dt: float):
  import mjlab.tasks  # noqa: F401  (registers the stock tasks)
  from mjlab.scene import Scene
  from mjlab.tasks.registry import load_env_cfg
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, "cpu").compile()
  m = cfg.sim.mujoco
  model.opt.timestep = dt
  model.opt.iterations, model.opt.ls_iterations = m.iterations, m.ls_iterations
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  return model, round(cfg.sim.mujoco.timestep * cfg.decimation / dt)


def fly(onnx: str, dt: float) -> dict:
  arrays = export.load(onnx)
  meta = json.loads(str(arrays["meta"]))
  model, every = build(dt)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  names = meta["joint_names"].split(",")
  default = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
  scale = np.array([float(v) for v in meta["action_scale"].split(",")])
  terms = meta["observation_names"].split(",")
  qadr = np.array([model.jnt_qposadr[model.joint(f"robot/{n}").id] for n in names])
  vadr = np.array([model.jnt_dofadr[model.joint(f"robot/{n}").id] for n in names])
  act = np.array([next(i for i in range(model.nu)
                       if model.actuator_trnid[i, 0] == model.joint(f"robot/{n}").id)
                  for n in names])
  data.qpos[qadr] = default
  mujoco.mj_forward(model, data)

  def sensor(name):
    s = model.sensor(f"robot/{name}")
    return data.sensordata[s.adr[0]:s.adr[0] + s.dim[0]]
  root = model.body("robot/pelvis").id
  last = np.zeros(len(names))
  rows, fell = [], None
  for seconds, vx, vy, wz in SEGMENTS:
    cmd = np.array([vx, vy, wz])
    steps = int(round(seconds / dt))
    got = []
    for k in range(steps):
      if k % every == 0:
        rot = data.xmat[root].reshape(3, 3)
        obs = {"base_lin_vel": lambda: sensor("imu_lin_vel"),
               "base_ang_vel": lambda: sensor("imu_ang_vel"),
               "projected_gravity": lambda: rot.T @ [0.0, 0.0, -1.0],
               "joint_pos": lambda: data.qpos[qadr] - default,
               "joint_vel": lambda: data.qvel[vadr],
               "actions": lambda: last,
               "command": lambda: cmd}
        x = np.concatenate([np.asarray(obs[t](), dtype=float) for t in terms])
        last = export.forward(arrays, x)
        data.ctrl[act] = default + scale * last
      mujoco.mj_step(model, data)
      if data.xpos[root][2] < 0.4 and fell is None:
        fell = data.time
      if k > steps // 3:
        rot = data.xmat[root].reshape(3, 3)
        v = rot.T @ data.qvel[0:3]          # the free joint's linear velocity is the world's
        got.append((v[0], v[1], data.qvel[5]))
    g = np.array(got).mean(axis=0)
    rows.append({"asked": [vx, vy, wz], "got": [round(float(v), 3) for v in g]})
  return {"dt": dt, "fell_at_s": fell, "segments": rows}


def main(argv=None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx")
  ap.add_argument("--dt", type=float, nargs="+", default=[0.005, 0.002])
  args = ap.parse_args(argv)
  arrays = export.load(args.onnx)
  err = export.check(args.onnx, arrays)
  print(f"numpy matches the ONNX file to {err:.1e}")
  for dt in args.dt:
    out = fly(args.onnx, dt)
    print(f"dt {dt}: {'FELL at %.1f s' % out['fell_at_s'] if out['fell_at_s'] else 'stood throughout'}")
    for r in out["segments"]:
      a, g = r["asked"], r["got"]
      print(f"  asked vx {a[0]:+.2f} vy {a[1]:+.2f} wz {a[2]:+.2f} -> "
            f"vx {g[0]:+.2f} vy {g[1]:+.2f} wz {g[2]:+.2f}")


if __name__ == "__main__":
  main()
