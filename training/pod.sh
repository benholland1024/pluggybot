#!/bin/bash
# Rent a pod -- a GPU to train on (#377), or CPUs to fly a batch of sim
# flights on (#469) -- and bring the results home: on Vast.ai, or on Runpod
# with PROVIDER=runpod (#510; training/README.md has what each costs). A pod's
# id says whose it is (a Vast instance's is a number, a Runpod pod's letters
# and digits), so only the two creates read PROVIDER, and the SSH half is one.
#
#   training/pod.sh create NAME "GPU A,GPU B,..."   rent one, print its id: on Vast the cheapest offer that passes and starts
#   training/pod.sh create-cpu NAME [VCPUS]          rent CPUs: VCPUS (32) on a cheap GPU's machine
#   training/pod.sh address POD                      its public "HOST PORT" for SSH, once up
#   training/pod.sh stop POD | delete POD            stop (the disk stays, and bills) / delete
#   training/pod.sh setup  HOST PORT     copy training/ + the body's files, install, check the GPU
#   training/pod.sh train  HOST PORT TASK ENVS ITERS RUN [extra mjlab flags...]
#   training/pod.sh status HOST PORT RUN
#   training/pod.sh pull   HOST PORT     runs/ back to training/runs/ (gitignored)
#
#   training/pod.sh setup-sim HOST PORT [REF]         the repo at REF (HEAD), installed, a frame rendered
#   training/pod.sh batch HOST PORT NAME CMD...        CMD from the repo's root, detached
#   training/pod.sh batch-status HOST PORT NAME        its meta, its log's tail, the pod's share in use
#   training/pod.sh pull-batch HOST PORT NAME          back to $BATCHES/NAME (~/pluggybot-training/batches)
#
# A batch of flights is CPU work -- MuJoCo steps on one core a process and the
# cameras render in software (osmesa, as the serving image does) -- and never
# flies bare on the dev box (CLAUDE.md says why). Its pod has no volume, so a
# batch lives on the container disk: pull it, then delete.
#
# HOST and PORT are the pod's public SSH address (`pod.sh address POD`).
# The key is the one `runpodctl doctor` made, on both providers: a Runpod pod
# gets its public half as PUBLIC_KEY at creation, which Runpod's images copy
# into authorized_keys; the Vast account holds it (its Keys page) and puts it
# in every instance created after it was added, never in one before.
# The code and its environment go on the pod's local disk (/root); the runs go
# on /workspace. On Runpod /root is fast and gone on a stop, and /workspace is
# the volume a stop keeps (a network filesystem on the secure cloud, slow for
# a venv's many small files). On Vast both are the instance's one disk, which
# a stop keeps but whose GPU it lets go: a stopped instance runs again only
# once that GPU is free, so pull before a stop. Delete once the runs are home.
set -euo pipefail
cmd=$1
provider=${PROVIDER:-vast}
[[ $provider == vast || $provider == runpod ]] || { echo "PROVIDER is vast or runpod, not $provider" >&2; exit 2; }

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
# Vast.ai's go through its CLI (`uv tool install vastai==1.8.3`), which reads
# its API key from its own config (~/.config/vastai/vast_api_key).
vast_call() {
  python3 - "$@" <<'PY'
import json, subprocess, sys, time

# Vast's own image, so its SSH launch mode works, without CUDA: the venv
# brings its own (1.3 GB against the Runpod image's 10.6).
IMAGE = "vastai/base-image:stock-ubuntu24.04-py312-2026-09-07"
DISK, CPU_DISK = "50", "20"
#: How long a rented instance may take to start, s: its image pulled at the
#: host's speed, and the container made.
START_S = 600

def vast(*args):
  """The CLI's answer as JSON (None if it gave none) and everything it said.
  ⚠ It exits 0 when it is refused, so its exit code says nothing."""
  try:
    r = subprocess.run(["vastai", *args, "--raw"], capture_output=True, text=True)
  except FileNotFoundError:
    sys.exit("no vastai on the PATH: uv tool install vastai==1.8.3")
  said = (r.stdout + r.stderr).strip()
  try:
    return json.loads(r.stdout), said
  except ValueError:
    return None, said

def started(i):
  """None once instance i runs with its SSH port mapped, else why it never
  will. ⚠ A host can fail to make the container at all (2026-10-16: a 4090's
  "unresolvable CDI devices"), and Vast then stops what we never asked it to."""
  t0 = time.time()
  while time.time() - t0 < START_S:
    d, _ = vast("show", "instance", i)
    if isinstance(d, dict):
      if d.get("actual_status") == "running" and (d.get("ports") or {}).get("22/tcp"):
        return None
      if d.get("actual_status") in ("exited", "unknown", "offline") or d.get("intended_status") == "stopped":
        return (d.get("status_msg") or str(d.get("actual_status"))).strip()[:200]
    time.sleep(10)
  return "not running after %d s" % START_S

def destroyed(i):
  """Whether instance i is gone. `destroy instance` says nothing under --raw,
  so the answer is its absence."""
  vast("destroy", "instance", i, "-y")
  for _ in range(10):
    d, _ = vast("show", "instance", i)
    if isinstance(d, dict) and "instances" in d and d["instances"] is None:
      return True
    time.sleep(3)
  return False

def key_on_account(pub):
  """An instance gets the account's keys as it is created, never after."""
  keys, said = vast("show", "ssh-keys")
  if keys is None:
    sys.exit("the account's SSH keys unread: " + said[:200])
  if pub.split()[1] not in json.dumps(keys):
    sys.exit("the key is not on the Vast account: add the .pub on https://cloud.vast.ai/manage-keys/")

def search(query, disk):
  offers, said = vast("search", "offers", query, "--storage", disk, "-o", "dph+")
  if not isinstance(offers, list):
    sys.exit("no offers read: " + said[:400])
  if not offers:
    sys.exit("no offer passes: " + query)
  return offers

def rent(offers, disk, name, what):
  """The first of the offers, in their order, that rents and starts, printed
  with what(offer). An offer can go between the search and the rental, and
  --cancel-unavail refuses rather than creating a stopped instance that bills
  its disk; one that never starts is destroyed, and the next tried."""
  for o in offers[:5]:
    made, said = vast("create", "instance", str(o["id"]), "--image", IMAGE, "--disk", disk,
                      "--ssh", "--direct", "--cancel-unavail", "--label", name)
    if not (isinstance(made, dict) and made.get("new_contract")):
      print("offer %s not rented: %s" % (o["id"], said[:200]), file=sys.stderr)
      continue
    i = str(made["new_contract"])
    print("rented %s (offer %s), waiting for it to start" % (i, o["id"]), file=sys.stderr)
    why = started(i)
    if why is None:
      # the price is the machine's and its disk's; traffic is billed apart, by the byte
      print(i, "|", what(o), "|", "%.3f $/h with %s GB" % (o["dph_total"], disk),
            "| traffic $%.0f in, $%.0f out a TB |" % (1000 * o["inet_down_cost"], 1000 * o["inet_up_cost"]),
            o.get("geolocation"), "| reliability %.3f" % o["reliability"])
      return
    print("%s never ran (%s): %s" % (i, why, "destroyed" if destroyed(i) else "NOT destroyed, delete it"),
          file=sys.stderr)
  sys.exit("refused: none of the five cheapest offers ran")

# Every offer taken is on-demand, never interruptible: a reclaimed host loses
# the run. A verified host at a reliability of 0.99 or more, in a data centre
# or not (Ben, 2026-10-16: that day the one data-centre 4090 was $1.35/h, the
# cheapest other $0.47); a direct SSH port; and a day's rental at least, since
# an instance past its host's end date stops and is deleted 48 h later.
TERMS = "num_gpus=1 verified=true rentable=true reliability>=0.99 direct_port_count>=1 duration>=1"

cmd, args = sys.argv[1], sys.argv[2:]
if cmd == "create":
  name, gpus, pub = args
  key_on_account(pub)
  # Vast names a card without its maker: Runpod's "NVIDIA GeForce RTX 4090" is its "RTX 4090".
  cards = [g.strip().removeprefix("NVIDIA ").removeprefix("GeForce ").replace(" ", "_")
           for g in gpus.split(",")]
  # a driver for CUDA 12.8 or later, as `setup`'s torch builds need
  offers = search("gpu_name in [%s] %s cuda_vers>=12.8 disk_space>=%s cpu_ram>=16"
                  % (",".join(cards), TERMS, DISK), DISK)
  rent(sorted(offers, key=lambda o: o["dph_total"]), DISK, name, lambda o: o["gpu_name"])
elif cmd == "create-cpu":
  name, vcpus, pub = args
  key_on_account(pub)
  # Any card's machine with the vCPUs asked for and 2 GB a vCPU, as Runpod's is
  # asked; the cheapest machine first, never the cheapest vCPU-hour, which
  # rented a whole 256-vCPU EPYC with 1 GB a vCPU for a batch of 32 at $1.06/h.
  offers = search("%s cpu_cores_effective>=%s cpu_ram>=%d disk_space>=%s"
                  % (TERMS, vcpus, 2 * int(vcpus), CPU_DISK), CPU_DISK)
  rent(sorted(offers, key=lambda o: o["dph_total"]), CPU_DISK, name,
       lambda o: "%.0f vCPUs, %.0f GB, %s" % (o["cpu_cores_effective"], o["cpu_ram"] / 1000,
                                              (o.get("cpu_name") or "?").strip()))
elif cmd == "address":
  d, said = vast("show", "instance", args[0])
  if not isinstance(d, dict) or not d.get("id"):
    sys.exit("no instance %s: %s" % (args[0], said[:200]))
  ip = (d.get("public_ipaddr") or "").strip()
  port = ((d.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
  if d.get("actual_status") != "running" or not ip or not port:
    sys.exit("not up yet: %s %s" % (d.get("actual_status"), (d.get("status_msg") or "").strip()[:200]))
  print(ip, port)
elif cmd == "stop":
  # `stop instance` answers in words, --raw or not
  _, said = vast("stop", "instance", args[0])
  if "stopping instance" not in said:
    sys.exit("refused: " + said[:400])
  print("stopped", args[0])
elif cmd == "delete":
  if not destroyed(args[0]):
    sys.exit("refused: %s is still there" % args[0])
  print("deleted", args[0])
PY
}
on_vast() { [[ $1 =~ ^[0-9]+$ ]]; }
case $cmd in
  create)
    name=$2 gpus=$3
    pub=$(cat "${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}.pub")
    if [ "$provider" = vast ]; then vast_call create "$name" "$gpus" "$pub"; exit 0; fi
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
    if on_vast "$2"; then vast_call address "$2"; exit 0; fi
    api GET "/pods/$2" | python3 -c "
import json,sys; d=json.load(sys.stdin)
ip, port = d.get('publicIp'), (d.get('portMappings') or {}).get('22')
if not ip or not port: sys.exit('not up yet: ' + str(d.get('desiredStatus')) + ' ' + str(d.get('lastStatusChange')))
print(ip, port)"
    exit 0 ;;
  create-cpu)
    # The CPUs of a cheap GPU's machine, never a CPU pod: on 2026-10-06 every
    # CPU pod over 2 vCPUs was refused while the catalog said HIGH, and one
    # RTX 3090's share came with 32 vCPUs and 125 GB at $0.50/h, half the
    # 32-vCPU CPU pod's price. The cheapest GPU first.
    name=$2 vcpus=${3:-32}
    pub=$(cat "${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}.pub")
    if [ "$provider" = vast ]; then vast_call create-cpu "$name" "$vcpus" "$pub"; exit 0; fi
    body=$(python3 - "$name" "$vcpus" "$pub" <<'PY'
import json, sys
name, vcpus, pub = sys.argv[1], int(sys.argv[2]), sys.argv[3]
print(json.dumps({
  "name": name, "computeType": "GPU", "gpuCount": 1, "gpuTypePriority": "custom",
  "gpuTypeIds": ["NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA A40", "NVIDIA L4",
                 "NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4090"],
  "minVCPUPerGPU": vcpus, "minRAMPerGPU": 2 * vcpus, "cloudType": "SECURE",
  "imageName": "runpod/base:1.4.0-ubuntu2404",
  "containerDiskInGb": 20, "volumeInGb": 0,
  "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub},
}))
PY
)
    api POST /pods "$body" | python3 -c "
import json,sys; d=json.load(sys.stdin)
if 'id' not in d: sys.exit('refused: ' + json.dumps(d)[:400])
print(d['id'], '|', d.get('vcpuCount'), 'vCPUs', d.get('memoryInGb'), 'GB |', d.get('costPerHr'), '\$/h')"
    exit 0 ;;
  stop)
    if on_vast "$2"; then vast_call stop "$2"; exit 0; fi
    api POST "/pods/$2/stop" >/dev/null && echo "stopped $2"; exit 0 ;;
  delete)
    if on_vast "$2"; then vast_call delete "$2"; exit 0; fi
    api DELETE "/pods/$2" >/dev/null && echo "deleted $2"; exit 0 ;;
esac

host=$2 port=$3
shift 3
key=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh_=(ssh -i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "root@$host")
remote=/root/pluggybot
# What a pod may use is its cgroup's: `nproc`, `free` and the load are the
# HOST's (a 32-vCPU share read 256 CPUs and 1007 GB). Vast's is a hard quota,
# so a batch takes one job fewer: 30 workers on a 30.7-CPU share ran each 12 %
# slower than 16 did.
share_py='
def read(*paths):
  for p in paths:
    try:
      return open(p).read().split()
    except OSError:
      pass
  return []
def gb(v):
  return f"{int(v[0]) / 2**30:.0f}" if v and v[0].isdigit() else "?"
cpu = read("/sys/fs/cgroup/cpu.max") or (read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
                                         + read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
cpus = f"{int(cpu[0]) / int(cpu[1]):.1f}" if len(cpu) == 2 and cpu[0].isdigit() else "unlimited"
used = read("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
cap = read("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
print(f"the pod: {cpus} CPUs, {gb(used)} of {gb(cap)} GB in use")'
share="python3 -c $(printf %q "$share_py")"

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
    # A batch still flying runs from this tree: replaced under it, its code
    # and the COMMIT its meta.txt names would no longer be what it flew.
    "${ssh_[@]}" 'for m in /root/batches/*/meta.txt; do
      [ -f "$m" ] && ! grep -q "^exit " "$m" && { echo "refused: ${m%/meta.txt} has not exited" >&2; exit 1; }
    done; exit 0'
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
echo "$(.venv/bin/python -V), commit $(cat COMMIT)"
# the house as the spikes build it, and one frame off the colour imager
MUJOCO_GL=osmesa .venv/bin/python -c "
import mujoco
from pluggybot.legs import world as lw
m = lw.home_spec().compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 72, 128); r.update_scene(d, 'color_eye')
print('the house built, a frame rendered', r.render().shape); r.close()"
EOF
    "${ssh_[@]}" "$share"
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
    "${ssh_[@]}" "cat /root/batches/$name/meta.txt; tail -4 /root/batches/$name/log.txt; $share"
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
