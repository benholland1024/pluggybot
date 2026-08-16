# The PluggyWorld serving image (rooftop-media-2026 #20).
#
# Runs `scripts/serve.py`: the battery-driven hub lifecycle, headless, paced
# to real time, publishing protocol frames + grid PNGs + event lines to the
# website's ingest socket as an outbound WebSocket CLIENT. The container
# therefore exposes no port and needs no inbound rule -- if the website is
# down, the robot keeps living and frames drop (docs/Webserver.md).
#
# Rendering is osmesa, on the CPU. The serving box has no GPU and does not
# want one: the loop renders ~3 tag views per sim-second, which software
# rendering carries above 1x real time on four cores (the measurement in
# docs/Webserver.md).
#
# Build from the repo root:  docker build -t pluggyworld-sim .

FROM python:3.12-slim

# libosmesa6 is the entire GL story here -- MuJoCo's osmesa backend dlopens
# libOSMesa.so.8 when the first Renderer is built, and nothing else in this
# image touches a display. (A missing one is not a build error: it surfaces
# as an exception minutes into the first mission, which is why the smoke
# test below renders before the image is called good.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 \
 && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY deploy/requirements-serve.txt deploy/
RUN pip install --no-cache-dir -r deploy/requirements-serve.txt

# Smoke-test headless GL at BUILD time: one offscreen render of a one-geom
# world. It costs a second and turns "the deploy is broken" into "the build
# is red" -- the acceptance criterion "headless GL verified on the actual
# server" is then just building (or running) this image there.
RUN python -c "import mujoco; \
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>'); \
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); \
print('osmesa render ok', r.render().shape)"

# Model paths in `world_config()` are CWD-relative, so the world only loads
# from /app. models/ carries the tag textures the cameras see, and
# home_world.meta.json -- the sidecar the boards are described in.
COPY models/ models/
COPY src/ src/
COPY scripts/serve.py scripts/
COPY deploy/entrypoint.sh deploy/

# Board contents, the points ledger and the overseer's memory are WORLD state,
# not run state (issues #12, #14 and #15): mount a volume here and the
# whiteboards, the balance the site puts on its scoreboard, and the robot's
# journal all survive the restart that ends every mission. `goals.md` lives in
# the same volume and is the one file a HUMAN writes -- editing it changes what
# the robot is for, with no redeploy and no code change.
#
# The user gets a real home directory: mesa writes its shader cache there,
# and without one every osmesa context logs "Failed to create /home/pluggy
# for shader cache ---disabling" and recompiles shaders from scratch.
RUN mkdir -p /var/lib/pluggybot \
 && useradd --system --uid 10001 --create-home --home-dir /home/pluggy pluggy \
 && chown pluggy /var/lib/pluggybot
USER pluggy
ENV HOME=/home/pluggy \
    PLUGGY_BOARDS=/var/lib/pluggybot/boards.json \
    PLUGGY_LEDGER=/var/lib/pluggybot/ledger.json \
    PLUGGY_JOURNAL=/var/lib/pluggybot/journal.json \
    PLUGGY_GOALS=/var/lib/pluggybot/goals.md

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
