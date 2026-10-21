"""One attempt in a world of its own, flown and streamed (issue #530, stage
2): the child process a session's run is, and the flight a recording of one
is made by.

  fly     the attempt: the task's set-out built afresh (`attempts.set_out`),
          its stream opened -- the header, the set-out's scene -- the skill
          run inside the task's limit while the grader watches every step,
          the call, the hold, one verdict through `scoring.evaluate`, and
          the record's row; the stream ends with the attempt's own message
  main    the child: the job on stdin, the session's own token in its
          environment, the stream to the session's own ingest path; once
          graded, its word to the host in the file the host named and the
          row appended to the record, both before the stream's last message

A person's procedure is checked against the hands' vocabulary the robot's
practice flies (`steps.hand_facts`), refused with every reason before
anything is built, and run as the robot's are (`evaluation.practice`): the
same verbs, budgets and reports, graded by the same door. What a page shows
of the robot's senses is read on streams of the watcher's own
(`Body.hand_watched`, `Body.depth_watched`), so watching never moves it.
"""

from __future__ import annotations

import json
import os
import sys
import time

from pluggybot.activity.base import ActivitySet
from pluggybot.evaluation import attempts as at
from pluggybot.evaluation import counter as ct
from pluggybot.evaluation.practice import ProcedureLife, procedure_skill
from pluggybot.procedure import axes, lang
from pluggybot.procedure.steps import Refused, hand_facts
from pluggybot.session import wire
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The world a session's stream names (`evaluation.counter.WORLD`).
WORLD = ct.WORLD
#: A world of its own has no pack; its record says full, as the chest
#: demo's does (`scripts/imagine_chest.py`), because a frame's robot record
#: carries one.
FULL = {"frac": 1.0, "watts": 0.0, "charging": False}
#: The child waits this long, wall s, for its stream to open before it
#: flies: an attempt nobody can watch is not flown.
CONNECT_S = 10.0
#: ...and this long, wall s, for its last word to leave.
FLUSH_S = 5.0
#: Where the child finds the session's own token: never a flag (`ps` is
#: public), and never the host's shared secret, which it is not handed.
TOKEN_ENV = "PLUGGY_SESSION_TOKEN"


def facts():
  """What a person's procedure is checked against and runs in: the hands'
  vocabulary, `measure` with it, as the robot's practice flies it."""
  return hand_facts(measure=True)


def task_of(chest: int) -> at.Task:
  """The task a session flies: the lid of chest `chest`, the humanoid at
  its counter (the attempt kit's)."""
  return at.Task(chest=chest, body="humanoid", mount="counter")


class WatchedLife(ProcedureLife):
  """A procedure's life in a session: the practice's (`ProcedureLife`),
  each verb's start and every line the run says sent to the stream."""

  def __init__(self, body, send) -> None:
    super().__init__(body)
    self._send = send

  def verb_began(self, n: int, line: int, verb: str, args: dict) -> None:
    self._send(wire.step_message(float(self.data.time), n, line, verb, args))

  def _say(self, text: str, detail: str = "", **kw) -> None:
    super()._say(text, detail, **kw)
    self._send({"type": "event", "t": round(float(self.data.time), 3), "robot": ROBOT_ROOT,
                "line": text})


class Watch:
  """The robot's senses as its procedure reads them (`read`), sent each
  `wire.READOUT_S`: its hands' places on the watcher's own encoders
  (`Body.hand_watched`), and everything else off what the body has already
  measured -- a force, a gripper's word, a verb's report -- None where a
  sense has no reading yet."""

  PLACE = ("x", "y", "z")

  def __init__(self, life, sensors, send) -> None:
    self.life, self.sensors, self._send = life, tuple(sensors), send
    self._next = 0.0

  def read(self) -> dict:
    out: dict = {}
    places: dict = {}
    for name in self.sensors:
      side, _, key = name.partition(".")
      if side in axes.HANDS and key in self.PLACE:
        if side not in places:
          places[side] = axes.in_stance(self.life, self.life.body.hand_watched(side))
        out[name] = places[side][self.PLACE.index(key)]
        continue
      try:
        out[name] = float(axes.SENSORS[name].read(self.life))
      except axes.NoReading:
        out[name] = None
    return out

  def step_hook(self) -> None:
    t = float(self.life.data.time)
    if t < self._next:
      return
    self._next = t + wire.READOUT_S
    self._send(wire.readouts_message(t, self.read()))


def sender(sink):
  """How a typed message reaches a sink: a publisher's `message`, a
  recorder's `emit`."""
  return getattr(sink, "message", None) or sink.emit


class Flight:
  """What `fly` leaves behind: the row, the run's record, the stream's last
  message, and the world it flew in -- its compiled model and its scene."""

  def __init__(self) -> None:
    self.row: dict | None = None
    self.ran: dict | None = None
    self.message: dict | None = None
    self.model = None
    self.scene: dict | None = None


def last_word(sink, msg: dict) -> bool:
  """The stream's last message: a publisher's waits up to `FLUSH_S` for room
  rather than be dropped (`WsPublisher.message(wait=)`); a recorder's never
  is. False: it was dropped all the same."""
  message = getattr(sink, "message", None)
  if message is None:
    sink.emit(msg)
    return True
  return message(msg, wait=FLUSH_S) is not False


def fly(task: at.Task, open_sink, *, proc: lang.Procedure | None = None, skill=None,
        n: int = 1, pace: float | None = None, info: dict | None = None,
        watch: bool = True, build=None, hold_s: float | None = None,
        budget_s: float | None = None, on_row=None) -> Flight:
  """One attempt at `task`, streamed into the sink `open_sink(model, data,
  **stream)` opens once the set-out is built (`build`, `attempts.set_out`
  unless a test's): a person's `proc` -- or, for a recording, a code
  `skill(body)` -- run inside the task's limit, graded, the verdict sent as
  the stream's last word. `pace` holds sim time to wall time (a live
  session's 1.0); `watch` sends the robot's senses; `hold_s` and
  `budget_s` are the grader's hold and the try's limit, the task's own
  unless a test's. `on_row(row, ran)` is handed the graded row BEFORE the
  last word goes: a child killed after the verdict has kept it. Raises
  what the attempt raised (`attempts.Unstable` among it), having said so on
  the stream."""
  from pluggybot.mind.practice import run_report
  from pluggybot.telemetry.pacer import RealTimePacer
  if (proc is None) == (skill is None):
    raise ValueError("an attempt flies a procedure or a skill: one of the two")
  if pace is not None and not pace > 0:
    raise ValueError(f"pace is sim seconds a wall second, over 0: {pace}")
  facts_ = facts()
  world = (build or at.set_out)(task)
  flight = Flight()
  said = {"state": "SET_OUT", "status": "set out"}
  body = world.body
  sink, added = None, []

  def status() -> dict:
    return {**said, "posture": body.posture, "battery": FULL}
  try:
    acts = ActivitySet([a for a in (world.chest, world.activity) if a is not None])
    flight.model = world.model
    flight.scene = ct.scene(world.model,
                            {"head": body.head_camera} if body.head_camera else None)
    stream = {"model_name": WORLD, "status_fn": status, "activities": acts,
              "opening": [wire.scene_message(flight.scene)],
              "samplers": [wire.DepthSampler(body)] if watch else []}
    sink = open_sink(world.model, world.data, **stream)
    send = sender(sink)
    life = WatchedLife(body, send)
    hooks = [sink.step_hook]
    if watch:
      hooks.append(Watch(life, facts_.sensors, send).step_hook)
    if pace is not None:
      hooks.append(RealTimePacer(world.data, rate=pace).step_hook)
    body.step_hooks.extend(hooks)
    added = hooks
    out: dict = {}
    inner = procedure_skill(proc, facts_, out, life=life) if proc is not None else skill

    def attempt_skill(b):
      said.update(state="ATTEMPT", status="trying")
      try:
        return (yield from inner(b))
      finally:
        # the hold follows the try however it ended, its time cut it too
        said.update(state="HOLD", status="the call: held still while the grader waits")
    try:
      row = at.attempt_in(world, attempt_skill, n, info=dict(info or {}),
                          **({} if hold_s is None else {"hold_s": hold_s}),
                          **({} if budget_s is None else {"budget_s": budget_s}))
    except Exception as e:
      flight.message = {"type": wire.ATTEMPT, "t": round(float(world.data.time), 3),
                        "robot": ROBOT_ROOT, "n": n, "task": task.as_dict(),
                        "error": f"{type(e).__name__}: {e}"}
      last_word(sink, flight.message)
      raise
    row["steps"] = out.get("steps", row.get("steps"))
    flight.row, flight.ran = row, out.get("ran")
    said.update(state="GRADED", status=("PASSED: " if row["verdict"]["ok"] else "failed: ")
                + row["verdict"]["reason"])
    report = (run_report(n, flight.ran, row["verdict"], row["simS"], row["cut"])
              if proc is not None else None)
    flight.message = wire.attempt_message(row, flight.ran, report, float(world.data.time))
    if on_row is not None:
      on_row(row, flight.ran)
    if not last_word(sink, flight.message):
      print("session: the stream's verdict was dropped (its queue stayed full); the host's "
            "`ended` carries it", file=sys.stderr, flush=True)
    return flight
  finally:
    try:
      for hook in added:
        body.step_hooks.remove(hook)
      if sink is not None:
        # a publisher's last word leaves before its socket closes (`flush`);
        # a recorder drains its queue on close
        flush = getattr(sink, "flush", None)
        if flush is not None and not flush(FLUSH_S):
          print(f"session: the stream's last words had not left after {FLUSH_S:.0f} s",
                file=sys.stderr, flush=True)
        sink.close()
    finally:
      world.close()


# ---- the child: one run, as the host starts it ---------------------------------------


def _wait_connected(publisher, seconds: float) -> bool:
  deadline = time.monotonic() + seconds
  while publisher.connections == 0 and time.monotonic() < deadline:
    time.sleep(0.05)
  return publisher.connections > 0


def run_job(job: dict, token: str, *, wait_s: float = CONNECT_S) -> tuple[int, dict]:
  """One run as the host hands it over (`wire.parse_run`'s fields, the
  endpoint joined, `result` the file the host reads): `(exit code, what to
  say)` -- 0 graded, 2 refused, 3 no stream, 1 failed. A graded run's word
  is written to `job["result"]`, and its row appended to `job["record"]`'s
  `rows.jsonl`, the moment it is graded: before the stream's last message,
  so a cancel or a kill that lands after it keeps both."""
  from pathlib import Path
  from pluggybot.evaluation.identity import repo_commit
  from pluggybot.telemetry.publisher import WsPublisher
  try:
    proc = lang.compile_procedure(job["source"], facts())
  except Refused as no:
    return 2, {"refused": list(no.reasons)}
  task = task_of(int(job["chest"]))

  def open_sink(model, data, **stream):
    pub = WsPublisher(model, data, job["endpoint"], token=token, **stream)
    if not _wait_connected(pub, wait_s):
      pub.close()
      raise ConnectionError(f"the session's stream never opened ({pub.last_error})")
    return pub
  graded: dict = {}

  def keep(row: dict, ran: dict | None) -> None:
    graded.update(ok=row["verdict"]["ok"], reason=row["verdict"]["reason"])
    say(job, graded)
    if not job.get("record"):
      return
    kept = dict(row) | {"session": job["session"], "source": job["source"],
                        "run": row["run"] | {"world": WORLD}}
    try:
      path = Path(job["record"])
      path.mkdir(parents=True, exist_ok=True)
      with open(path / "rows.jsonl", "a") as f:
        f.write(json.dumps(kept, default=str) + "\n")
    except OSError as e:
      # the verdict still goes: a record that cannot be written is said
      print(f"session: the row was not kept ({e})", file=sys.stderr, flush=True)
  info = {"process": "person", "skill": proc.name, "tokens": 0, "commit": repo_commit()}
  try:
    fly(task, open_sink, proc=proc, n=int(job["attempt"]), pace=job.get("pace", 1.0),
        info=info, on_row=keep)
  except ConnectionError as e:
    return 3, {"why": str(e)}
  except Exception as e:  # noqa: BLE001 -- said on the stream; the host says it ended
    if graded:
      # graded, and then its stream's close raised: the verdict stands
      return 0, graded
    return 1, {"why": f"{type(e).__name__}: {e}"}
  return 0, graded


def say(job: dict, said: dict) -> None:
  """The child's one word to its host, in the file the host named
  (`job["result"]`), whole or not at all (a kill mid-write leaves the old
  one): never stdout, which the container's log shares with MuJoCo's
  warnings."""
  if not job.get("result"):
    return
  part = job["result"] + ".part"
  with open(part, "w") as f:
    json.dump(said, f)
  os.replace(part, job["result"])


def main() -> None:
  """The child: the job on stdin as JSON, the token in `TOKEN_ENV`; its
  word in the job's `result` file (`say`), and the exit code `run_job`'s."""
  job = json.loads(sys.stdin.read())
  token = os.environ.get(TOKEN_ENV, "")
  if not token:
    say(job, {"why": f"no {TOKEN_ENV}"})
    raise SystemExit(1)
  code, said = run_job(job, token)
  if code != 0:
    say(job, said)
  raise SystemExit(code)


if __name__ == "__main__":
  main()
