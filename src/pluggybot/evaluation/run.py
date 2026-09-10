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

from pluggybot.evaluation.arms import DEFAULT_RUNG, RUNGS, arm_flags  # noqa: F401
from pluggybot.evaluation.record import Probe, build_record, data_hashes, validate

# `arm_flags` and `RUNGS` moved to `evaluation/arms.py` in issue #142, when
# the DEPLOYED world became able to fly an arm as well: two definitions of
# what an arm means is how a stream ends up claiming an arm nobody flew.
# Re-exported here because this is where every caller has always found them.


def run_config(config: dict, out: Path, partial: Path | None = None) -> dict:
  from pluggybot.lifecycle import run_demo, world_config
  from pluggybot.mind import overseer as ov

  flags = arm_flags(config["arm"], config.get("rung") or DEFAULT_RUNG)
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
  hashes = data_hashes(config["world"])
  cfg = world_config(config["world"])
  config = {**config,
            "packWh": (cfg["battery_wh"] if config["pack"] == "demo"
                       else cfg["hosting_battery_wh"]),
            "reserveWh": cfg["low_battery_wh"],
            "deadlineS": (ov.CALL_TIMEOUT_S if flags["overseer"] else None),
            # The dead robot's restart timer (issue #143), recorded whether
            # or not it is on: a run whose robot could stand itself up is a
            # different experiment from one whose robot could not, and a
            # field that only appeared when the feature was used would make
            # every older record ambiguous rather than negative.
            "restartAfterS": config.get("restartAfterS")}
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
      # ⚠ MORTAL, and the harness is the one caller that must be: deaths
      # split by cause are what the arms are judged on (Evaluation.md §3),
      # and a run that quietly survived a flat pack would report a
      # survival span that never happened.
      mortal=True,
      # ...and DELIBERATELY NOT auto-restarting (issue #143). A measured run
      # is about ONE life. `survival.survivalS` is already a list, so several
      # spans per run are representable -- but the rollup's survival
      # statistics were written against one span per run, and turning this on
      # by default would change what every committed number means without
      # anybody choosing it. The served world is where it belongs; if the
      # harness ever wants it, that is a deliberate change to the rollup in
      # the same breath. `config["restartAfterS"]` records the None.
      restart_after_s=config.get("restartAfterS"),
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
