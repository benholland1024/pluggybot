#!/bin/bash
# Rent a Runpod pod -- a GPU to train on (#377), or CPUs to fly a batch of
# sim flights on (#469) -- and bring the results home.
#
#   training/pod.sh create NAME "GPU A,GPU B,..."   rent one (Runpod REST API), print its id
#   training/pod.sh create-cpu NAME [FLAVORS] [VCPUS]  rent CPUs (cpu3c,cpu5c; 32 vCPUs)
#   training/pod.sh address POD                      its public "HOST PORT" for SSH, once up
#   training/pod.sh stop POD | delete POD            stop (the volume stays, and bills) / delete
#   training/pod.sh setup  HOST PORT     copy training/ + the body's files, install, check the GPU
#   training/pod.sh train  HOST PORT TASK ENVS ITERS RUN [extra mjlab flags...]
#   training/pod.sh status HOST PORT RUN
#   training/pod.sh pull   HOST PORT     runs/ back to training/runs/ (gitignored)
#
#   training/pod.sh setup-sim HOST PORT [REF]         the repo at REF (HEAD), installed, a frame rendered
#   training/pod.sh batch HOST PORT NAME CMD...        CMD from the repo's root, detached
#   training/pod.sh batch-status HOST PORT NAME        its meta, its log's tail, the load and memory
#   training/pod.sh pull-batch HOST PORT NAME          back to $BATCHES/NAME (~/pluggybot-training/batches)
#
# A batch of flights is CPU work -- MuJoCo steps on one core a process and the
# cameras render in software (osmesa, as the serving image does) -- and it is
# never flown on the dev box beside its desktop: six processes there took the
# box into swap and its desktop down (2026-10-06). A CPU pod has no pod
# volume, so a batch lives on the container disk: pull it, then delete.
#
# HOST and PORT are the pod's public SSH address (`pod.sh address POD`).
# The key is the one `runpodctl doctor` made; the pod gets its public half as
# PUBLIC_KEY at creation, which Runpod's images copy into authorized_keys.
# The code and its environment go on the pod's local disk (/root: fast, gone
# on a stop); the runs go on /workspace, the volume a stop keeps (a network
# filesystem on Runpod's secure cloud, slow for a venv's many small files).
set -euo pipefail
cmd=$1

# The pod lifecycle goes through Runpod's REST API: `runpodctl pod create`
# uses the GraphQL API, which a key scoped to pods (REST read/write, GraphQL
# read) is refused on. The key is read from runpodctl's config, never shown.
api() {
  local method=$1 path=$2 body=${3:-}
  local key
  key=$(python3 -c "import tomllib,os; print(tomllib.load(open(os.path.expanduser('~/.runpod/config.toml'),'rb'))['apikey'])")
  curl -sS -X "$method" "https://rest.runpod.io/v1$path" -H "Authorization: Bearer $key" \
    -H "Content-Type: application/json" ${body:+-d "$body"}
}
case $cmd in
  create)
    name=$2 gpus=$3
    pub=$(cat "${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}.pub")
    # CLOUD=SECURE for Runpod's own data centres; CUDA lists the host drivers
    # accepted (a 12.x host gets torch's cu128 build at `setup`).
    body=$(python3 - "$name" "$gpus" "$pub" "${CLOUD:-COMMUNITY}" "${CUDA:-12.8,12.9,13.0}" <<'PY'
import json, sys
name, gpus, pub, cloud, cuda = sys.argv[1:6]
print(json.dumps({
  "name": name, "computeType": "GPU", "gpuCount": 1,
  "gpuTypeIds": gpus.split(","), "gpuTypePriority": "custom",
  "cloudType": cloud, "supportPublicIp": True,
  "allowedCudaVersions": cuda.split(","),
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "containerDiskInGb": 30, "volumeInGb": 20, "volumeMountPath": "/workspace",
  "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub},
}))
PY
)
    api POST /pods "$body" | python3 -c "
import json,sys; d=json.load(sys.stdin)
if 'id' not in d: sys.exit('refused: ' + json.dumps(d)[:400])
print(d['id'], '|', d.get('gpu',{}).get('displayName') or d.get('gpu'), '|', d.get('costPerHr'), '\$/h')"
    exit 0 ;;
  address)
    api GET "/pods/$2" | python3 -c "
import json,sys; d=json.load(sys.stdin)
ip, port = d.get('publicIp'), (d.get('portMappings') or {}).get('22')
if not ip or not port: sys.exit('not up yet: ' + str(d.get('desiredStatus')) + ' ' + str(d.get('lastStatusChange')))
print(ip, port)"
    exit 0 ;;
  create-cpu)
    name=$2 flavors=${3:-cpu3c,cpu5c} vcpus=${4:-32}
    pub=$(cat "${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}.pub")
    # cpu3c and cpu5c carry 2 GB a vCPU: 32 hold ~28 flights at ~1.2 GB each.
    body=$(python3 - "$name" "$flavors" "$vcpus" "$pub" <<'PY'
import json, sys
name, flavors, vcpus, pub = sys.argv[1:5]
print(json.dumps({
  "name": name, "computeType": "CPU", "cpuFlavorIds": flavors.split(","),
  "cpuFlavorPriority": "custom", "vcpuCount": int(vcpus), "cloudType": "SECURE",
  "imageName": "runpod/base:1.4.0-ubuntu2404",
  "containerDiskInGb": 30, "volumeInGb": 0,
  "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub},
}))
PY
)
    api POST /pods "$body" | python3 -c "
import json,sys; d=json.load(sys.stdin)
if 'id' not in d: sys.exit('refused: ' + json.dumps(d)[:400])
print(d['id'], '|', d.get('vcpuCount'), 'vCPUs', d.get('memoryInGb'), 'GB |', d.get('costPerHr'), '\$/h')"
    exit 0 ;;
  stop) api POST "/pods/$2/stop" >/dev/null && echo "stopped $2"; exit 0 ;;
  delete) api DELETE "/pods/$2" >/dev/null && echo "deleted $2"; exit 0 ;;
esac

host=$2 port=$3
shift 3
key=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh_=(ssh -i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "root@$host")
remote=/root/pluggybot

case $cmd in
  setup)
    "${ssh_[@]}" "mkdir -p $remote/models $remote/training /workspace/runs"
    tar -C "$repo" -czf - --exclude=.venv --exclude=runs --exclude=MUJOCO_LOG.TXT \
      --exclude=__pycache__ training models/quadruped.xml models/quadruped.json \
      | "${ssh_[@]}" "tar -C $remote -xzf -"
    "${ssh_[@]}" bash -s <<'EOF'
set -e
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# `import mujoco` loads libEGL when MUJOCO_GL=egl (mjlab sets it); training
# never renders, but the library must be there to import.
ldconfig -p | grep -q libEGL.so.1 || { apt-get update -qq && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 >/dev/null; }
cd /root/pluggybot/training
uv sync -q
# torch 2.14 from PyPI is built for CUDA 13; on a host whose driver stops at
# 12.x, the same version's cu128 build goes in its place.
driver_cuda=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
if [ "${driver_cuda:-13}" -lt 13 ]; then
  torch_v=$(.venv/bin/python -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])")
  uv pip install -q --python .venv/bin/python --reinstall "torch==$torch_v" \
    --index-url https://download.pytorch.org/whl/cu128
fi
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
cd /root/pluggybot/training
export WARP_CACHE_PATH=/root/.warp-cache
# MuJoCo Warp warns on every step a box lies on a height field with more than
# 50 contacts (a fallen body on rough ground): a stairs run wrote 1.5 M such
# lines, 258 MB to the network volume, and slowed to a third of its speed.
PYTHONUNBUFFERED=1 nohup bash -c '"$@" 2>&1 | grep --line-buffered -v \
  -e "height field collision overflow" -e "decrease the number of hfield"' _ \
  .venv/bin/python -m quad_train.train "$task" --env.scene.num-envs "$envs" \
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
  setup-sim)
    # The commit itself, never the working tree: what flies is what REF
    # names, and its id goes into every batch's meta.txt.
    ref=${1:-HEAD}
    commit=$(git -C "$repo" rev-parse --short "$ref")
    "${ssh_[@]}" "rm -rf $remote && mkdir -p $remote /root/batches"
    git -C "$repo" archive --format=tar "$ref" \
      | "${ssh_[@]}" "tar -C $remote -xf - && echo $commit > $remote/COMMIT"
    "${ssh_[@]}" bash -s <<'EOF'
set -e
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# MuJoCo's osmesa backend dlopens libOSMesa when the first Renderer is built
ldconfig -p | grep -q libOSMesa.so.8 || { apt-get update -qq && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libosmesa6 >/dev/null; }
cd /root/pluggybot
uv sync -q --frozen
echo "$(nproc) vCPUs, $(free -g | awk '/Mem/{print $2}') GB, $(.venv/bin/python -V), commit $(cat COMMIT)"
# the house as the spikes build it, and one frame off the colour imager
MUJOCO_GL=osmesa .venv/bin/python -c "
import mujoco
from pluggybot.legs import world as lw
m = lw.home_spec().compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 72, 128); r.update_scene(d, 'color_eye')
print('the house built, a frame rendered', r.render().shape); r.close()"
EOF
    ;;
  batch)
    # The remote shell splits the command line again, so each word goes quoted.
    "${ssh_[@]}" "bash -s -- $(printf '%q ' "$@")" <<'EOF'
set -e
name=$1; shift
out=/root/batches/$name
mkdir -p "$out"
cd /root/pluggybot
printf '%s\n' "commit $(cat COMMIT)" "started $(date -Is)" "cmd $*" > "$out/meta.txt"
# one BLAS thread and one llvmpipe thread a process, as a batch is a process
# a core (llvmpipe draws the same pixels on one thread as on all of them)
OUT=$out PATH=/root/pluggybot/.venv/bin:$HOME/.local/bin:$PATH MUJOCO_GL=osmesa \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONUNBUFFERED=1 nohup bash -c \
  '"$@" > "$OUT/log.txt" 2>&1; echo "exit $? at $(date -Is)" >> "$OUT/meta.txt"' _ "$@" \
  </dev/null >/dev/null 2>&1 &
echo "started $name: pid $!"
EOF
    ;;
  batch-status)
    name=$1
    "${ssh_[@]}" "cat /root/batches/$name/meta.txt; tail -4 /root/batches/$name/log.txt; uptime; free -g | head -2"
    ;;
  pull-batch)
    name=$1
    dest=${BATCHES:-$HOME/pluggybot-training/batches}
    mkdir -p "$dest"
    "${ssh_[@]}" "tar -C /root/batches -czf - $name" | tar -C "$dest" -xzf -
    ls -la "$dest/$name"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
