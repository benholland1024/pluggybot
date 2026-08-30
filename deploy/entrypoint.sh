#!/bin/sh
# env -> `serve.py` flags, so a compose file configures the sim with
# `environment:` alone and never restates a command line.
#
# The ingest secret is NOT among them: serve.py reads $PLUGGYWORLD_TOKEN
# itself, precisely so the secret never appears in `ps` (docs/Webserver.md).
#
# Anything passed to the container is appended AFTER the derived flags, and
# argparse takes the last occurrence -- so `docker run <image> --rate 2.0`
# overrides PLUGGY_RATE without the entrypoint knowing the flag exists.
set -eu

set -- --endpoint "${PLUGGY_ENDPOINT:-ws://localhost:8765}" \
       --world "${PLUGGY_WORLD:-home}" \
       --errand "${PLUGGY_ERRAND:-draw}" \
       --rate "${PLUGGY_RATE:-1.0}" \
       --max-sim-time "${PLUGGY_MAX_SIM_TIME:-3600}" \
       "$@"

if [ -n "${PLUGGY_BOARDS:-}" ]; then
  set -- --boards "${PLUGGY_BOARDS}" "$@"
fi
if [ -n "${PLUGGY_LEDGER:-}" ]; then
  set -- --ledger "${PLUGGY_LEDGER}" "$@"
fi
# Job offers (issue #21). A path implies the board is on, exactly as for the
# boards and the ledger, and for the same reason: /var/lib/pluggybot is a
# volume and an offer is world state, so a restart resumes the jobs the last
# mission left standing rather than re-offering them.
if [ -n "${PLUGGY_TASKS:-}" ]; then
  set -- --task-state "${PLUGGY_TASKS}" "$@"
fi
if [ -n "${PLUGGY_BATTERY_WH:-}" ]; then
  set -- --battery-wh "${PLUGGY_BATTERY_WH}" "$@"
fi
# Which CELL a watched world runs on (pluggybot #15). `hosting` is the named
# hours-long pack; PLUGGY_BATTERY_WH still overrides it with a raw number,
# which is what the deployment did before the name existed.
if [ -n "${PLUGGY_PACK:-}" ]; then
  set -- --pack "${PLUGGY_PACK}" "$@"
fi
if [ -n "${PLUGGY_RESERVE_WH:-}" ]; then
  set -- --reserve-wh "${PLUGGY_RESERVE_WH}" "$@"
fi

# The robot's MEMORY (issue #38), and deliberately NOT behind the overseer
# flag: History.md is written on every world, the documents stream on every
# world, and the site's Thoughts tab is what a visitor opens first. Two of
# the four files are a human's to edit on the volume; the sim refuses a
# write to any file by anyone but its owner.
if [ -n "${PLUGGY_THOUGHTS:-}" ]; then
  set -- --thoughts "${PLUGGY_THOUGHTS}" "$@"
fi
if [ -n "${PLUGGY_GOALS:-}" ]; then
  set -- --goals "${PLUGGY_GOALS}" "$@"
fi

# POINTS AS FOOD (issue #36) is environment too, and read straight from it by
# scripts/serve.py -- $PLUGGY_METABOLISM names a metabolism.json on the volume
# and IMPLIES the mechanic, the way $PLUGGY_TASKS implies a task board. So is
# every other tuning file the sim reads for itself: $PLUGGY_REWARDS (what a
# job pays), $PLUGGY_CADENCE (how busy the world is), $PLUGGY_QUESTIONS (the
# question bank) and $PLUGGY_ENERGY (what an errand costs). Mount a file, no
# rebuild, no flag here.
# ⚠ Unset means the robot never gets hungry and its balance is unbounded,
# which is what every world did before #36 -- turning hunger on in a
# deployment is a deliberate act, because the rate and the cap are still
# provisional numbers rather than measured ones.

# The LLM overseer (issue #15). Like the ingest secret, $ANTHROPIC_API_KEY
# and $HF_TOKEN are NOT turned into flags -- serve.py never sees them and the
# backends read them from the environment, so they stay out of `ps`. Which
# backend is $PLUGGY_OVERSEER_BACKEND's call, defaulting to $PLUGGY_MODEL's
# shape (an `org/name` id is the HuggingFace router, mind/llm.py; a bare id is
# Anthropic; `local` is a model on the box at $PLUGGY_OVERSEER_URL, issue
# #19). All of them are read by `hub.overseer.build` from the environment
# directly, like PLUGGY_OVERSEER and for the same reason -- $PLUGGY_OVERSEER_
# KEY has no business in `ps` either -- but the --overseer flag is passed
# anyway so that a run with it on says so in its own argv.
# The allowance and the operator's switch (issue #37) are environment too,
# and read straight from it by mind/spend.py and mind/mode.py -- no flags, so
# nothing here has to know they exist: $PLUGGY_WEEKLY_USD, $PLUGGY_SPEND,
# $PLUGGY_MODE_FILE, $PLUGGY_ESCALATE_TO. ⚠ The mode file and the spend file
# BOTH belong on the state volume: a budget that reset on restart would be no
# budget (a mission ends several times an hour here), and a robot paused
# before a restart must come back paused.
if [ -n "${PLUGGY_ESCALATE_TO:-}" ]; then
  set -- --escalate-to "${PLUGGY_ESCALATE_TO}" "$@"
fi
if [ -n "${PLUGGY_OVERSEER:-}" ] && [ "${PLUGGY_OVERSEER}" != "0" ]; then
  set -- --overseer "$@"
  if [ -n "${PLUGGY_JOURNAL:-}" ]; then
    set -- --journal "${PLUGGY_JOURNAL}" "$@"
  fi
  if [ -n "${PLUGGY_OVERSEER_BUDGET:-}" ]; then
    set -- --overseer-budget "${PLUGGY_OVERSEER_BUDGET}" "$@"
  fi
fi

exec python scripts/serve.py "$@"
