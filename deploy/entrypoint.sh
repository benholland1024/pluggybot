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
if [ -n "${PLUGGY_BATTERY_WH:-}" ]; then
  set -- --battery-wh "${PLUGGY_BATTERY_WH}" "$@"
fi

exec python scripts/serve.py "$@"
