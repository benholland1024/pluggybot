"""The session host (issue #530, stage 2): the long-lived process on the
server that the site hands a person's runs to, ONE AT A TIME (Ben,
2026-10-21: sessions run on the deploy box, admins only, one seat).

It dials OUT to the site's `/api/pluggyworld/sessions/host`, as the sim and
the eye do, with the shared secret (`$PLUGGYWORLD_TOKEN`, never a flag), and
says `hello` on every connect: its build, the task, the job as the robot is
told it and the text a procedure is written against -- the same text the
robot's mind is given, so the page's reference and the robot's never drift
apart. A `run` is a person's procedure for one attempt:

  - checked first, here, against the language (`lang.compile_procedure`):
    refused with every reason at once, and nothing built;
  - else flown by a CHILD PROCESS (`session.attempt`), a fresh one a run --
    a world dropped in a reused one is held until the cycle collector runs
    (#469) -- handed the job on stdin and the session's OWN token in its
    environment, never the shared secret nor any other credential; it
    streams to the session's own path, at real time, and once graded writes
    its word to a file the host named and the row to the record;
  - a second run while one is in flight is `busy`, a `cancel` stops it only
    when it names its session and attempt, and a child that outlives
    `CHILD_S` is killed and said to be.

A run is `graded` if its child wrote a verdict, whatever ended the process
after; else `cancelled` or `out of time` if the host stopped it; else
`failed`, saying why. A dropped connection drops nothing in flight: the
child streams on its own path and is watched meanwhile, the next `hello`
says what is running, and an ending not yet said follows it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

from pluggybot.challenge import lid
from pluggybot.evaluation import attempts as at
from pluggybot.procedure import lang
from pluggybot.procedure.steps import Refused
from pluggybot.session import wire
from pluggybot.session.attempt import TOKEN_ENV, facts, task_of

#: A child is killed past this, wall s: a try's limit (67.6 s at the
#: counter) and its hold paced at real time, the world's build and the
#: stream's open, many times over.
CHILD_S = 600.0
#: ...and after a cancel or a kill, given this long to end before SIGKILL.
STOP_S = 5.0
#: Between connection attempts, wall s.
RECONNECT_S = 2.0
#: How long a receive waits before the host looks at its child again, s.
POLL_S = 0.25
#: The shared secret the site's host path takes, and where the host is told
#: where the site is and where the record goes (the image's environment).
SECRET_ENV, ENDPOINT_ENV, RECORD_ENV = ("PLUGGYWORLD_TOKEN", "PLUGGY_SESSION_ENDPOINT",
                                        "PLUGGY_SESSION_RECORD")
#: The chests a session may fly: the public ones (`attempts.PUBLIC`), never
#: a held-out draw.
CHESTS = at.PUBLIC
#: What a child's environment never carries: the shared secret, and any
#: other credential the service holds, known by its name (the mind's keys,
#: `HF_TOKEN`, `PLUGGY_OVERSEER_KEY`: a person's run flies no mind).
CREDENTIAL = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|KEY", re.IGNORECASE)


def hello(build: dict, running: dict | None = None) -> dict:
  """The host's opening word: its build, the task, the job and the text a
  procedure is written against (`mind.practice.system_sections`, the
  language and the body) with its names as data (`vocabulary_index`), and
  the run in flight -- or ended and not yet said -- if any."""
  from pluggybot.evaluation.practice import job_for
  from pluggybot.mind.practice import system_sections, vocabulary_index
  task = task_of(CHESTS.start)
  sections = dict(system_sections(facts()))
  msg = wire.hello_message(
    build, {"grader": task.grader, "body": task.body, "mount": task.mount,
            "limitS": round(task.limit_s, 2), "holdS": lid.HOLD_S,
            "chests": [CHESTS.start, CHESTS.stop - 1]},
    job_for(task), [(h, sections[h]) for h in ("the language", "your body")],
    vocabulary_index(facts()), lang.MAX_SOURCE_CHARS)
  msg["running"] = running
  return msg


#: The vendored fixture (`--hello`): what a fresh host says, no build and
#: nothing running -- the site's guide for people is checked against it.
HELLO_FIXTURE = "protocol/hello.home_humanoid_counter.json"


def hello_fixture() -> str:
  return json.dumps(hello({}), indent=1, ensure_ascii=False) + "\n"


def child_env(token: str) -> dict:
  """A child's environment: the host's, LESS the shared secret and every
  other credential (`CREDENTIAL`), plus the session's own token -- a run
  reaches its own path and nothing else."""
  env = {k: v for k, v in os.environ.items()
         if k != SECRET_ENV and not CREDENTIAL.search(k)}
  env[TOKEN_ENV] = token
  return env


def spawn_child(job: dict, token: str):
  """A run's child process (`session.attempt`), the job on its stdin; its
  stdout and stderr are the host's (the container's log)."""
  proc = subprocess.Popen([sys.executable, "-m", "pluggybot.session.attempt"],
                          stdin=subprocess.PIPE, text=True, env=child_env(token))
  try:
    proc.stdin.write(json.dumps(job))
    proc.stdin.close()
  except OSError:
    proc.kill()
    proc.wait()
    raise
  return proc


def result_file() -> str:
  """A file for a child's word to its host (`attempt.say`)."""
  fd, path = tempfile.mkstemp(prefix="pluggy-run-", suffix=".json")
  os.close(fd)
  return path


def read_result(path: str | None) -> dict:
  """A child's word, and its file gone: {} where it wrote none, or a word
  that is not one."""
  if not path:
    return {}
  try:
    with open(path) as f:
      said = json.load(f)
  except (OSError, ValueError):
    said = {}
  finally:
    for gone in (path, path + ".part"):
      try:
        os.unlink(gone)
      except OSError:
        pass
  return said if isinstance(said, dict) else {}


class Host:
  """One host's state across its connections: the run in flight -- or
  ended and not yet said -- and its child. `spawn` is how a child starts (a
  test's own), `clock` the wall clock."""

  def __init__(self, endpoint: str, record: str | None = None, pace: float = 1.0,
               spawn=spawn_child, clock=time.monotonic) -> None:
    parts = urlsplit(endpoint)
    self.origin = f"{parts.scheme}://{parts.netloc}"
    self.record, self.pace = record, pace
    self.spawn, self.clock = spawn, clock
    self.run: dict | None = None
    self.child = None
    self.result: str | None = None
    self.started_at = 0.0
    self.stopping: str | None = None
    #: the run's `ended`, made once and kept until a connection takes it
    self.ending: dict | None = None

  def running(self) -> dict | None:
    return None if self.run is None else {"session": self.run["session"],
                                          "attempt": self.run["attempt"]}

  def handle(self, msg: dict, send) -> None:
    """A word from the site: a `run` or a `cancel`; anything else is
    dropped."""
    kind = msg.get("type")
    if kind == wire.RUN:
      self._run(msg, send)
    elif kind == wire.CANCEL:
      if self.run is not None and (msg.get("session"), msg.get("attempt")) == (
          self.run["session"], self.run["attempt"]):
        self._stop("cancelled")

  def _run(self, msg: dict, send) -> None:
    session, attempt = str(msg.get("session", "")), msg.get("attempt")
    if self.run is not None:
      send(wire.busy_message(session, attempt))
      return
    try:
      run = wire.parse_run(msg, CHESTS)
    except wire.BadRun as bad:
      send(wire.ended_message(session, attempt if isinstance(attempt, int) else 0,
                              "failed", str(bad)))
      return
    try:
      lang.compile_procedure(run["source"], facts())
    except Refused as no:
      send(wire.refused_message(run["session"], run["attempt"], no.reasons))
      return
    result = result_file()
    job = {"session": run["session"], "attempt": run["attempt"], "chest": run["chest"],
           "source": run["source"], "endpoint": self.origin + run["ingest"],
           "record": self.record, "pace": self.pace, "result": result}
    try:
      child = self.spawn(job, run["token"])
    except Exception as e:  # noqa: BLE001 -- a process that would not start is the run's end
      read_result(result)
      send(wire.ended_message(run["session"], run["attempt"], "failed",
                              f"its process did not start ({type(e).__name__}: {e})"))
      return
    self.child, self.result = child, result
    self.run, self.started_at, self.stopping, self.ending = run, self.clock(), None, None
    send(wire.started_message(run["session"], run["attempt"], run["chest"]))

  def _stop(self, why: str) -> None:
    """End the child: asked to stop, then killed past `STOP_S`."""
    if self.child is None or self.stopping is not None:
      return
    self.stopping = why
    self.child.terminate()
    try:
      self.child.wait(timeout=STOP_S)
    except subprocess.TimeoutExpired:
      self.child.kill()

  def poll(self, send=None) -> None:
    """Has the child ended? Then its `ended` is made, once, and said, and
    the seat is free -- or kept, with the run, while there is no connection
    (`send` None, or one that raises): the next `hello` names the run and
    its ending follows. One past `CHILD_S` is stopped first, connected or
    not."""
    if self.ending is None:
      if self.child is None:
        return
      if self.child.poll() is None:
        if self.clock() - self.started_at > CHILD_S:
          self._stop("out of time")
        else:
          return
      self.ending = self._ending(self.child.wait())
      self.child = self.result = None
    if send is None:
      return
    send(self.ending)
    self.run = self.ending = self.stopping = None

  def _ending(self, code: int) -> dict:
    """How the run ended: graded where its child wrote a verdict, whatever
    ended the process after it (a cancel that raced a clean exit, a kill as
    the stream closed); else the host's own stop; else failed."""
    said = read_result(self.result)
    session, attempt = self.run["session"], self.run["attempt"]
    if isinstance(said.get("ok"), bool):
      return wire.ended_message(session, attempt, "graded",
                                verdict={"ok": said["ok"], "reason": said.get("reason", "")})
    if self.stopping is not None:
      return wire.ended_message(session, attempt, self.stopping)
    why = said.get("why") or "; ".join(said.get("refused") or ()) or f"exit {code}"
    return wire.ended_message(session, attempt, "failed", str(why))


def serve(endpoint: str, token: str, record: str | None, pace: float = 1.0,
          build: dict | None = None, *, host: Host | None = None,
          stop: threading.Event | None = None, reconnect_s: float = RECONNECT_S) -> None:
  """Connect, say hello, take runs; reconnect for ever (or until `stop`).
  The run in flight carries on across a dropped connection, watched
  meanwhile (`Host.poll`), and its ending is said after the next hello."""
  from websockets.sync.client import connect
  host = host or Host(endpoint, record, pace)
  stop = stop or threading.Event()
  build = build or {}
  while not stop.is_set():
    try:
      with connect(endpoint, open_timeout=5.0,
                   additional_headers={"Authorization": f"Bearer {token}"}) as ws:
        def send(msg: dict) -> None:
          ws.send(json.dumps(msg, separators=(",", ":")))
        send(hello(build, host.running()))
        print(f"session host: connected to {endpoint}", flush=True)
        while not stop.is_set():
          host.poll(send)
          try:
            raw = ws.recv(timeout=POLL_S)
          except TimeoutError:
            continue
          try:
            msg = json.loads(raw)
          except ValueError:
            continue
          if isinstance(msg, dict):
            host.handle(msg, send)
    except Exception as e:  # noqa: BLE001 -- refused, reset, a handshake the site answered
      # 401 or 503 (`InvalidStatus` is no OSError): all "no connection now"
      print(f"session host: no connection ({type(e).__name__}: {e}); again in "
            f"{reconnect_s:.0f} s", flush=True)
      deadline = time.monotonic() + reconnect_s
      while not stop.is_set() and time.monotonic() < deadline:
        host.poll(None)
        stop.wait(POLL_S)


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--endpoint", default=os.environ.get(ENDPOINT_ENV, ""),
                  help=f"the site's host path, ws://...; ${ENDPOINT_ENV}")
  ap.add_argument("--record", default=os.environ.get(RECORD_ENV) or None,
                  help=f"where each run's row is appended (rows.jsonl); ${RECORD_ENV}")
  ap.add_argument("--pace", type=float, default=1.0,
                  help="sim seconds a wall second (1.0: what a person watches)")
  ap.add_argument("--hello", nargs="?", const=HELLO_FIXTURE, metavar="PATH",
                  help=f"write the hello a fresh host says to PATH (default {HELLO_FIXTURE}, "
                       "the fixture the site vendors) and exit")
  args = ap.parse_args(argv)
  if args.hello:
    with open(args.hello, "w") as f:
      f.write(hello_fixture())
    print(f"{args.hello}: the hello, {os.path.getsize(args.hello):,} bytes")
    return
  if not args.endpoint:
    ap.error(f"no endpoint: --endpoint or ${ENDPOINT_ENV}")
  if not args.pace > 0:
    ap.error("--pace is sim seconds a wall second, over 0")
  token = os.environ.get(SECRET_ENV, "")
  if not token.strip():
    ap.error(f"no ${SECRET_ENV}: the site's host path takes the shared secret")
  from pluggybot.evaluation.identity import repo_commit
  serve(args.endpoint, token, args.record, args.pace, {"commit": repo_commit()})


if __name__ == "__main__":
  main()
