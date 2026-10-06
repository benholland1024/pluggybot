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

import io
import json
import os
import subprocess
import sys

import numpy as np

from pluggybot.imagination.record import Record
from pluggybot.imagination.rollout import SETTLE_S, Diverged, Readings
from pluggybot.imagination.scene import Refused, dump, parse

HEADER = "__header__"
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


def handle(blob: bytes, seed: int) -> bytes | None:
  """One request's answer, or None for `stop`. ⚠ EVERY FAILURE IS AN ANSWER:
  one the worker did not catch ended it for every request after (a document
  of 24 piled parts overflowed MuJoCo's arena: `FatalError`)."""
  from pluggybot.imagination.compile import compile_scene
  from pluggybot.imagination.rollout import rollout
  try:
    header, arrays = unpack(blob)
  except Exception as e:                                  # noqa: BLE001
    return pack({"ok": False, "error": f"unreadable: {type(e).__name__}: {e}"})
  kind = header.get("kind")
  if kind == "stop":
    return None
  if kind != "rollout":
    return pack({"ok": False, "error": f"no request {kind!r}"})
  try:
    scene = parse(header["document"])
    record = Record.from_wire(header["record"], arrays)
    world = compile_scene(scene, carrying=record.start.carrying)
    out = rollout(world, record, settle_s=float(header.get("settle_s", SETTLE_S)),
                  noise_seed=seed if header.get("noisy") else None)
  except Refused as e:
    return pack({"ok": False, "refused": e.reasons})
  except Diverged as e:
    return pack({"ok": False, "diverged": str(e)})
  except Exception as e:                                  # noqa: BLE001
    return pack({"ok": False, "error": f"{type(e).__name__}: {e}"})
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
  while (blob := read_frame(inp)) is not None:
    answer = handle(blob, args.seed)
    if answer is None:
      return
    write_frame(out, answer)


class Imagination:
  """A worker process and the requests it serves: `rollout(document,
  record)`."""

  def __init__(self, seed: int = 0) -> None:
    self.seed = int(seed)
    self.closed = False
    self.process = subprocess.Popen(
      [sys.executable, "-m", "pluggybot.imagination.worker", "--seed", str(self.seed)],
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, env={**os.environ, **ONE_THREAD})

  def _ask(self, blob: bytes) -> tuple[dict, dict]:
    """One request and its answer. ⚠ A request cut off half way -- an
    exception between sending it and reading its whole answer, an
    interrupt -- ENDS THE WORKER: its answer would be left in the pipe, and
    every request after it would read the one before's (the review of #473:
    one behind, silently, or a hang once both outgrew the pipe)."""
    if self.closed:
      raise RuntimeError("this imagination is closed")
    if self.process.poll() is not None:
      raise RuntimeError(f"the worker has ended ({self.process.returncode})")
    try:
      write_frame(self.process.stdin, blob)
      answer = read_frame(self.process.stdout)
    except BrokenPipeError:
      self._end()
      raise RuntimeError(f"the worker has ended ({self.process.returncode})") from None
    except BaseException:
      self._end()
      raise
    if answer is None:
      self._end()
      raise RuntimeError(f"the worker ended without an answer ({self.process.returncode})")
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
      if "refused" in header:
        raise Refused(header["refused"])
      if "diverged" in header:
        raise Diverged(header["diverged"])
      raise RuntimeError(header.get("error", "the worker did not say"))
    readings = Readings.from_wire(header, out)
    if len(readings.t) != record.n:
      self._end()
      raise RuntimeError(f"the worker answered {len(readings.t)} rows to a record of "
                         f"{record.n}: it is out of step, and ended")
    return readings

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


if __name__ == "__main__":
  main()
