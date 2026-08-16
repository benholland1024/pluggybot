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
if [ -n "${PLUGGY_BATTERY_WH:-}" ]; then
  set -- --battery-wh "${PLUGGY_BATTERY_WH}" "$@"
fi

# The LLM overseer (issue #15). Like the ingest secret, $ANTHROPIC_API_KEY is
# NOT turned into a flag -- serve.py never sees it and the SDK reads it from
# the environment, so it stays out of `ps`. PLUGGY_OVERSEER is read by
# `hub.overseer.build` directly, but the flag is passed anyway so that a run
# with it on says so in its own argv.
if [ -n "${PLUGGY_OVERSEER:-}" ] && [ "${PLUGGY_OVERSEER}" != "0" ]; then
  set -- --overseer "$@"
  if [ -n "${PLUGGY_GOALS:-}" ]; then
    set -- --goals "${PLUGGY_GOALS}" "$@"
  fi
  if [ -n "${PLUGGY_JOURNAL:-}" ]; then
    set -- --journal "${PLUGGY_JOURNAL}" "$@"
  fi
  if [ -n "${PLUGGY_OVERSEER_BUDGET:-}" ]; then
    set -- --overseer-budget "${PLUGGY_OVERSEER_BUDGET}" "$@"
  fi
fi

exec python scripts/serve.py "$@"
