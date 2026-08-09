"""Guards for the camera dashboard (viz.py, issue #1)."""

import numpy as np

from pluggybot.viz import SEAM, TILE_H, TILE_W, ViewDashboard


def test_dashboard_composite_layout(playground_model, playground_data):
  import mujoco
  mujoco.mj_forward(playground_model, playground_data)
  dash = ViewDashboard(playground_model)
  try:
    frame = dash.render(playground_data)
  finally:
    dash.close()

  assert frame.shape == (2 * TILE_H + SEAM, 2 * TILE_W + SEAM, 3)
  left = frame[:TILE_H, :TILE_W]
  right = frame[:TILE_H, TILE_W + SEAM:]
  dock = frame[TILE_H + SEAM:, TILE_W + SEAM:]
  assert left.std() > 0, "left-eye tile is blank"
  assert dock.std() > 0, "dock-eye tile is blank"
  assert not np.array_equal(left, right), "stereo tiles show no parallax"


def test_dashboard_embeds_map_tile(playground_model, playground_data):
  import mujoco
  mujoco.mj_forward(playground_model, playground_data)
  dash = ViewDashboard(playground_model)
  try:
    blank = dash.render(playground_data, map_img=None)
    # A recognizable map: white field with a black block off-center.
    map_img = np.full((200, 200), 255, dtype=np.uint8)
    map_img[40:60, 40:60] = 0
    with_map = dash.render(playground_data, map_img=map_img)
  finally:
    dash.close()

  map_tile_blank = blank[TILE_H + SEAM:, :TILE_W]
  map_tile = with_map[TILE_H + SEAM:, :TILE_W]
  assert not np.array_equal(map_tile, map_tile_blank), "map tile ignored map_img"
  # The map must arrive un-mirrored: the black block sits in the tile's
  # upper-left quadrant, as it does in the source image.
  h, w = map_tile.shape[:2]
  upper_left = map_tile[:h // 2, :w // 2].mean()
  lower_right = map_tile[h // 2:, w // 2:].mean()
  assert upper_left < lower_right, "map tile flipped or misplaced"


def test_battery_glyph_renders_and_reflects_state(playground_model, playground_data):
  import mujoco
  mujoco.mj_forward(playground_model, playground_data)
  dash = ViewDashboard(playground_model)
  try:
    plain = dash.render(playground_data)
    full = dash.render(playground_data, battery=0.9)
    low = dash.render(playground_data, battery=0.1)
    charging = dash.render(playground_data, battery=0.1, charging=True)
  finally:
    dash.close()
  region = np.s_[TILE_H + SEAM:TILE_H + SEAM + 24, TILE_W - 100:TILE_W]
  assert not np.array_equal(plain[region], full[region]), "glyph missing"
  assert not np.array_equal(full[region], low[region]), \
    "urgency color must change with charge level"
  assert not np.array_equal(low[region], charging[region]), \
    "charging must be visually distinct"
