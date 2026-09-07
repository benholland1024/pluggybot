"""One run of one configuration, in its own process (issue #106).

    python -m pluggybot.evaluation.run CONFIG.json OUT.json [PARTIAL.jsonl]

`scripts/experiment.py` spawns one of these per run, for two reasons that
are both about honesty rather than convenience: a run that wedges (#108)
can be killed on wall clock and recorded as `killed` from the rows it had
flushed to PARTIAL, and N runs share no process state -- the solver policy
note in CLAUDE.md is about exactly the kind of global that leaks between
missions in one interpreter.

The configuration is the tuple Evaluation.md §4 names, plus what the
baseline found mattered: `(world, arm, pack, model, seed)` and the errand,
whether tasks and hunger are on, the day length, and whether state is fresh.
"""

import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from pluggybot.evaluation.record import (
  BUILT_ARMS, Probe, build_record, data_hashes, validate,
)


def arm_flags(arm: str) -> dict:
  """What an arm means to `run_demo`. `autonomous` does not exist yet
  (Evaluation.md §7, item 4) and is refused rather than silently run as
  `guarded` -- a record claiming an arm that was not flown is the worst
  kind of result."""
  if arm == "scripted":
    return {"overseer": False}
  if arm == "guarded":
    return {"overseer": True}
  raise NotImplementedError(
    f"arm {arm!r} is not built; the built arms are {BUILT_ARMS} "
    "(docs/Evaluation.md §7, item 4 -- the `autonomous` arm is one branch "
    "in HubLifecycle.run(), after the harness and the reset exist)")


def run_config(config: dict, out: Path, partial: Path | None = None) -> dict:
  from pluggybot.lifecycle import run_demo, world_config
  from pluggybot.mind import overseer as ov

  flags = arm_flags(config["arm"])
  started = datetime.now(timezone.utc)
  sink_file = open(partial, "w") if partial is not None else None

  def sink(row: dict) -> None:
    if sink_file is not None:
      sink_file.write(json.dumps(row) + "\n")
      sink_file.flush()

  probe = Probe(sink)
  # Fresh state per run unless told otherwise: six fresh starts and six
  # consecutive days on one volume are different experiments (pass 1a).
  state = Path(config["stateDir"]) if config.get("stateDir") else Path(
    tempfile.mkdtemp(prefix="pluggy-run-"))
  state.mkdir(parents=True, exist_ok=True)
  hashes = data_hashes()
  cfg = world_config(config["world"])
  config = {**config,
            "packWh": (cfg["battery_wh"] if config["pack"] == "demo"
                       else cfg["hosting_battery_wh"]),
            "reserveWh": cfg["low_battery_wh"],
            "deadlineS": (ov.CALL_TIMEOUT_S if flags["overseer"] else None)}
  t0 = time.time()
  result = None
  try:
    result = run_demo(
      view=False, realtime=False, world=config["world"],
      pack=config["pack"], errand=config.get("errand", "draw"),
      max_sim_time=float(config.get("maxSimS", 3600.0)),
      tasks=bool(config.get("tasks", True)),
      metabolism=bool(config.get("metabolism", True)),
      overseer_model=config.get("model") if flags["overseer"] else None,
      overseer_backend=config.get("backend") if flags["overseer"] else None,
      thoughts_root=str(state / "thoughts"),
      ledger_state=str(state / "ledger.json"),
      board_state=str(state / "boards.json"),
      tasks_state=str(state / "tasks.json"),
      journal_state=str(state / "journal.json"),
      spend_state=str(state / "spend.json"),
      record=config.get("telemetry") or None,
      on_ready=probe.attach, **flags)
  finally:
    wall = time.time() - t0
    if sink_file is not None:
      sink_file.close()
    if not config.get("keepState"):
      shutil.rmtree(state, ignore_errors=True)
  record = validate(build_record(config, result, probe.events, wall, started,
                                 hashes=hashes))
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(record, indent=1) + "\n")
  return record


def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  if len(argv) < 2:
    print(__doc__)
    return 2
  config = json.loads(Path(argv[0]).read_text())
  partial = Path(argv[2]) if len(argv) > 2 else None
  record = run_config(config, Path(argv[1]), partial)
  print(f"{record['runId']}: {record['end']} at t={record['simSeconds']:.0f} "
        f"in {record['wallSeconds']:.0f} s, {record['mind']['decisions']} "
        f"decisions, voluntary charges "
        f"{record['charging']['voluntary']['chosen']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
