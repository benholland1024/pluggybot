#!/bin/bash
# Train on a rented GPU (a Runpod pod) and bring the results home (#377).
#
#   training/pod.sh setup  HOST PORT     copy training/ + the body's files, install, check the GPU
#   training/pod.sh train  HOST PORT TASK ENVS ITERS RUN [extra mjlab flags...]
#   training/pod.sh status HOST PORT RUN
#   training/pod.sh pull   HOST PORT     runs/ back to training/runs/ (gitignored)
#
# HOST and PORT are the pod's public SSH address (`runpodctl pod get <id>`).
# The key is the one `runpodctl doctor` made; the pod gets its public half as
# PUBLIC_KEY at creation, which Runpod's images copy into authorized_keys.
# Everything on the pod lives under /workspace, the volume a stop keeps.
set -euo pipefail
cmd=$1 host=$2 port=$3
shift 3
key=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh_=(ssh -i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "root@$host")
remote=/workspace/pluggybot

case $cmd in
  setup)
    "${ssh_[@]}" "mkdir -p $remote/models $remote/training /workspace/runs"
    tar -C "$repo" -czf - --exclude=.venv --exclude=runs --exclude=MUJOCO_LOG.TXT \
      --exclude=__pycache__ training models/quadruped.xml models/quadruped.json \
      | "${ssh_[@]}" "tar -C $remote -xzf -"
    "${ssh_[@]}" bash -s <<'EOF'
set -e
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/workspace/.uv-cache
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/pluggybot/training
uv sync -q
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c "import torch, mujoco, warp as wp; wp.init(); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mujoco', mujoco.__version__, '| warp', wp.config.version)"
EOF
    ;;
  train)
    task=$1 envs=$2 iters=$3 run=$4
    shift 4
    "${ssh_[@]}" bash -s -- "$task" "$envs" "$iters" "$run" "$@" <<'EOF'
set -e
task=$1 envs=$2 iters=$3 run=$4; shift 4
cd /workspace/pluggybot/training
export WARP_CACHE_PATH=/workspace/.warp-cache
nohup .venv/bin/python -m quad_train.train "$task" --env.scene.num-envs "$envs" \
  --agent.max-iterations "$iters" --agent.logger tensorboard --agent.run-name "$run" \
  --log-root /workspace/runs "$@" > "/workspace/runs/$run.log" 2>&1 &
echo "started $run: pid $!"
EOF
    ;;
  status)
    run=$1
    "${ssh_[@]}" "grep -E 'Learning iteration|Mean episode length|error_vel_xy|Iteration time|Traceback|Error' /workspace/runs/$run.log | tail -6; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
    ;;
  pull)
    mkdir -p "$repo/training/runs"
    "${ssh_[@]}" "tar -C /workspace -czf - runs" | tar -C "$repo/training" -xzf -
    ls "$repo/training/runs"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
