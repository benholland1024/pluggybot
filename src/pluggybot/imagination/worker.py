"""The imagination in a process of its own (issue #466): the codebase's first
worker process, a program of its own (`python -m
pluggybot.imagination.worker`). The robot lies still while it thinks, and
the served physics thread never steps an imagined world (the workshop's rig,
which does, cost a build 0.76 s there).

⚠ THE PROCESS BOUNDARY IS THE FENCE'S FIRST HALF: a request is a scene
document and a record, and nothing else crosses -- every message is a JSON
header and named numeric arrays in one `.npz`, read back with pickling off
(`pack`, `unpack`), framed on the worker's stdin and stdout. An `MjModel`,
an `MjData`, or anything else a pickle could carry, cannot be sent: the
worker builds its world from the document alone, and it is started from
nothing of the caller's (a program, not a fork).

Deterministic given its seed: a rollout steps no randomness of its own, and
the drivers' noise it may add (`rollout.rollout(noise_seed=)`) is keyed on
the worker's seed. A model in progress is never saved with the world: it is
built again.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from pluggybot.imagination.record import Record
from pluggybot.imagination.rollout import SETTLE_S, Diverged, Readings
from pluggybot.imagination.scene import Refused, dump, parse

HEADER = "__header__"


class Unbuildable(RuntimeError):
  """A document whose world could not be built or stepped (MuJoCo raised:
  an arena overflowed, a compile failed): the document's failure, which a
  fit and a round answer as they answer a world gone unstable -- never the
  worker's, which serves on."""
#: One thread a worker: a BLAS's own threads only fight the physics for the
#: cores (`quad_spike.py`: six doubled a step), and llvmpipe's would start
#: one a core.
ONE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "LP_NUM_THREADS": "1"}


def pack(header: dict, arrays: dict[str, np.ndarray] | None = None) -> bytes:
  """One message: `header` as JSON, `arrays` as numbers. Refuses anything
  else -- a type JSON cannot write, an array of objects -- by raising."""
  arrays = arrays or {}
  text = json.dumps(header, allow_nan=False)
  for name, a in arrays.items():
    if name == HEADER or not isinstance(a, np.ndarray) or a.dtype.kind not in "fiub":
      raise TypeError(f"{name!r} is not an array of numbers: a message carries "
                      f"a JSON header and numbers, nothing else")
  buf = io.BytesIO()
  np.savez(buf, **{HEADER: np.frombuffer(text.encode(), dtype=np.uint8)}, **arrays)
  return buf.getvalue()


def unpack(blob: bytes) -> tuple[dict, dict[str, np.ndarray]]:
  """`pack`'s message back: pickles refused (`allow_pickle=False`)."""
  with np.load(io.BytesIO(blob), allow_pickle=False) as z:
    header = json.loads(bytes(z[HEADER]).decode())
    return header, {k: z[k] for k in z.files if k != HEADER}


def write_frame(stream, blob: bytes) -> None:
  stream.write(len(blob).to_bytes(8, "little") + blob)
  stream.flush()


def read_frame(stream) -> bytes | None:
  """A frame, or None at the end of the stream."""
  head = stream.read(8)
  if len(head) < 8:
    return None
  n = int.from_bytes(head, "little")
  blob = stream.read(n)
  if len(blob) < n:
    return None
  return blob


#: The records a worker keeps for `residual` requests, the oldest let go.
KEPT_RECORDS = 4


def record_id(record: Record) -> str:
  """A record's name for `keep`: a hash of all it carries a rollout reads."""
  import hashlib
  head, arrays = record.to_wire()
  h = hashlib.sha256(json.dumps(head, sort_keys=True).encode())
  for name in sorted(arrays):
    h.update(name.encode() + np.ascontiguousarray(arrays[name]).tobytes())
  return h.hexdigest()[:20]


def handle(blob: bytes, seed: int, kept: dict | None = None) -> bytes | None:
  """One request's answer, or None for `stop`: a `rollout` (its readings),
  a `record` to keep (`KEPT_RECORDS`, in `kept`), or a `residual` -- a kept
  record's commands rolled out and answered as the binned force their
  readings put apart from the record's (`fit.residual`), all a fit's search
  reads. ⚠ EVERY FAILURE IS AN ANSWER: one the worker did not catch ended it
  for every request after (a document of 24 piled parts overflowed MuJoCo's
  arena: `FatalError`)."""
  from pluggybot.imagination.compile import compile_scene
  from pluggybot.imagination.rollout import rollout
  kept = {} if kept is None else kept
  try:
    header, arrays = unpack(blob)
  except Exception as e:                                  # noqa: BLE001
    return pack({"ok": False, "error": f"unreadable: {type(e).__name__}: {e}"})
  kind = header.get("kind")
  if kind == "stop":
    return None
  if kind not in ("rollout", "record", "residual"):
    return pack({"ok": False, "error": f"no request {kind!r}"})
  try:
    if kind == "record":
      kept[str(header["id"])] = Record.from_wire(header["record"], arrays)
      while len(kept) > KEPT_RECORDS:
        kept.pop(next(iter(kept)))
      return pack({"ok": True})
    if kind == "residual":
      record = kept.get(str(header["id"]))
      if record is None:
        return pack({"ok": False, "missing": str(header["id"])})
    else:
      record = Record.from_wire(header["record"], arrays)
    scene = parse(header["document"])
  except Refused as e:
    return pack({"ok": False, "refused": e.reasons})
  except Exception as e:                                  # noqa: BLE001
    return pack({"ok": False, "error": f"{type(e).__name__}: {e}"})
  # the document's own world: what fails building or stepping it is the
  # document's (`Unbuildable`), never the request's
  try:
    world = compile_scene(scene, carrying=record.start.carrying)
    out = rollout(world, record, settle_s=float(header.get("settle_s", SETTLE_S)),
                  noise_seed=seed if header.get("noisy") else None)
    if kind == "residual":
      from pluggybot.imagination.fit import residual
      a, b = (int(v) for v in header["rows"])
      return pack({"ok": True}, {"bins": residual(record.sensed, out, slice(a, b), record.dt)})
  except Diverged as e:
    return pack({"ok": False, "diverged": str(e)})
  except Exception as e:                                  # noqa: BLE001
    return pack({"ok": False, "unbuildable": f"{type(e).__name__}: {e}"})
  head, out_arrays = out.to_wire()
  return pack({"ok": True, **head}, out_arrays)


def main(argv=None) -> None:
  """The worker: frames in on stdin, answers out on what was stdout. ⚠ The
  C libraries print to file descriptor 1 (MuJoCo's warnings do), so it is
  pointed at stderr and the answers go out on a copy of it taken first."""
  import argparse
  ap = argparse.ArgumentParser(description="the imagination's worker (issue #466)")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)
  out = os.fdopen(os.dup(1), "wb")
  os.dup2(2, 1)
  sys.stdout = sys.stderr
  inp = sys.stdin.buffer
  kept: dict = {}
  while (blob := read_frame(inp)) is not None:
    answer = handle(blob, args.seed, kept)
    if answer is None:
      return
    write_frame(out, answer)


def _raise(header: dict) -> None:
  """A failed answer as what it was: refused, unstable, unbuildable -- the
  document's -- or the request's own error."""
  if "refused" in header:
    raise Refused(header["refused"])
  if "diverged" in header:
    raise Diverged(header["diverged"])
  if "unbuildable" in header:
    raise Unbuildable(header["unbuildable"])
  raise RuntimeError(header.get("error", "the worker did not say"))


def command(seed: int = 0) -> list[str]:
  """The worker's own command line, here."""
  return [sys.executable, "-m", "pluggybot.imagination.worker", "--seed", str(int(seed))]


class WorkerEnded(RuntimeError):
  """The worker's process ended -- killed, or the ssh connection it ran
  over dropped -- and the request with it: the connection's failure, never
  the document's."""


class Imagination:
  """A worker process and the requests it serves: `rollout(document,
  record)`, and `residual(document, record, rows)`. It is started by
  `argv` -- by default this machine's `command`; any command whose stdin
  and stdout are the worker's will do (an `ssh` to a pod's, the same frames
  over its pipe) -- and a command of the caller's sets its own threads."""

  def __init__(self, seed: int = 0, argv: list[str] | None = None) -> None:
    self.seed = int(seed)
    self.closed = False
    self.held: set[str] = set()
    env = None if argv is not None else {**os.environ, **ONE_THREAD}
    self.process = subprocess.Popen(argv or command(self.seed), stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, env=env)

  def _ask(self, blob: bytes) -> tuple[dict, dict]:
    """One request and its answer. ⚠ A request cut off half way -- an
    exception between sending it and reading its whole answer, an
    interrupt -- ENDS THE WORKER: its answer would be left in the pipe, and
    every request after it would read the one before's (the review of #473:
    one behind, silently, or a hang once both outgrew the pipe)."""
    if self.closed:
      raise RuntimeError("this imagination is closed")
    if self.process.poll() is not None:
      raise WorkerEnded(f"the worker has ended ({self.process.returncode})")
    try:
      write_frame(self.process.stdin, blob)
      answer = read_frame(self.process.stdout)
    except BrokenPipeError:
      self._end()
      raise WorkerEnded(f"the worker has ended ({self.process.returncode})") from None
    except BaseException:
      self._end()
      raise
    if answer is None:
      self._end()
      raise WorkerEnded(f"the worker ended without an answer ({self.process.returncode})")
    return unpack(answer)

  def rollout(self, document: dict, record: Record, settle_s: float = SETTLE_S,
              noisy: bool = False) -> Readings:
    """`record`'s commands replayed in the world `document` describes,
    worked out in the worker. The document is read here first, as the
    worker will read it, so a bad one is refused (`Refused`, every reason)
    the same in either place, and what is sent is its own canonical form.
    Raises `Diverged` for a world that went unstable, or `RuntimeError`
    with the worker's error."""
    canonical = dump(parse(document))
    head, arrays = record.to_wire()
    header, out = self._ask(pack({"kind": "rollout", "document": canonical, "record": head,
                                  "settle_s": float(settle_s), "noisy": bool(noisy)}, arrays))
    if not header.get("ok"):
      _raise(header)
    readings = Readings.from_wire(header, out)
    if len(readings.t) != record.n:
      self._end()
      raise RuntimeError(f"the worker answered {len(readings.t)} rows to a record of "
                         f"{record.n}: it is out of step, and ended")
    return readings

  def residual(self, document: dict, record: Record, rows: slice,
               settle_s: float = SETTLE_S) -> np.ndarray:
    """`fit.residual` of `record`'s rollout in `document`'s world over `rows`,
    worked out in the worker: the record is sent once and kept there
    (`KEPT_RECORDS`), sent again if the worker let it go. Raises as
    `rollout` does."""
    canonical = dump(parse(document))
    rid = record_id(record)
    for _ in range(2):
      if rid not in self.held:
        head, arrays = record.to_wire()
        header, _ = self._ask(pack({"kind": "record", "id": rid, "record": head}, arrays))
        if not header.get("ok"):
          raise RuntimeError(header.get("error", "the worker did not keep the record"))
        self.held.add(rid)
      header, out = self._ask(pack({"kind": "residual", "document": canonical, "id": rid,
                                    "rows": [rows.start or 0, rows.stop],
                                    "settle_s": float(settle_s)}))
      if header.get("missing"):
        self.held.discard(rid)
        continue
      if not header.get("ok"):
        _raise(header)
      return out["bins"]
    raise RuntimeError("the worker let the record go twice")

  def _end(self) -> None:
    if self.process.poll() is None:
      self.process.kill()
      self.process.wait()

  def close(self) -> None:
    """Asks the worker to stop, then lets go of it; a worker that has died,
    or a second call, is no error."""
    if self.closed:
      return
    self.closed = True
    if self.process.poll() is None:
      try:
        write_frame(self.process.stdin, pack({"kind": "stop"}))
      except OSError:
        pass
      try:
        self.process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        self._end()
    for stream in (self.process.stdin, self.process.stdout):
      try:
        stream.close()
      except OSError:
        pass

  def __enter__(self) -> "Imagination":
    return self

  def __exit__(self, *exc) -> None:
    self.close()


#: A worker that ended mid-request is started again and asked again this
#: many times, this long after (s): its answer is the document's and the
#: seed's, whoever serves it. ⚠ Every ssh connection to the pods dropped at
#: once, and took ten set-outs of #481's batch with it (#480's, eleven).
RESTARTS, RESTART_WAIT_S = 2, 5.0


class Imaginations:
  """`n` workers serving rollouts at once (`rollouts`): each request goes to
  whichever is free, and the answers come back in the order asked. Whoever
  served one, the answer is the same: a rollout steps no randomness of its
  own, and every worker has the one seed -- so a worker that ended is
  replaced and asked again (`RESTARTS`, counted in `restarts`)."""

  def __init__(self, n: int, seed: int = 0, argv: list[str] | None = None) -> None:
    if n < 1:
      raise ValueError(f"a pool has a worker or more, not {n}")
    self.seed, self.argv = seed, argv
    self.workers = [Imagination(seed, argv) for _ in range(n)]
    self.restarts = 0
    self._lock = threading.Lock()
    self._free: queue.SimpleQueue = queue.SimpleQueue()
    for w in self.workers:
      self._free.put(w)
    self._threads = ThreadPoolExecutor(n)

  @property
  def n(self) -> int:
    return len(self.workers)

  def _serve(self, ask):
    """`ask(worker)` on a free worker: its answer, or the `Refused`,
    `Diverged` or `Unbuildable` it was refused with; a worker that ended is
    replaced and asked again, and any other failure raises."""
    w = self._free.get()
    try:
      for tries in range(RESTARTS + 1):
        try:
          return ask(w)
        except (Refused, Diverged, Unbuildable) as e:
          return e
        except WorkerEnded:
          if tries == RESTARTS:
            raise
          time.sleep(RESTART_WAIT_S)
          w = self._replace(w)
    finally:
      self._free.put(w)

  def _replace(self, old: Imagination) -> Imagination:
    old.close()
    new = Imagination(self.seed, self.argv)
    with self._lock:
      self.workers[self.workers.index(old)] = new
      self.restarts += 1
    return new

  def rollouts(self, documents, record: Record, settle_s: float = SETTLE_S) -> list:
    """Each document's rollout of `record`, in order: its `Readings`, or the
    `Refused`, `Diverged` or `Unbuildable` it was refused with. Any other
    failure raises."""
    return list(self._threads.map(
      lambda d: self._serve(lambda w: w.rollout(d, record, settle_s)), list(documents)))

  def residuals(self, documents, record: Record, rows: slice,
                settle_s: float = SETTLE_S) -> list:
    """Each document's `Imagination.residual`, in order: its bins, or the
    `Refused`, `Diverged` or `Unbuildable` it was refused with. Any other
    failure raises."""
    return list(self._threads.map(
      lambda d: self._serve(lambda w: w.residual(d, record, rows, settle_s)),
      list(documents)))

  def close(self) -> None:
    self._threads.shutdown()
    for w in self.workers:
      w.close()

  def __enter__(self) -> "Imaginations":
    return self

  def __exit__(self, *exc) -> None:
    self.close()


if __name__ == "__main__":
  main()
