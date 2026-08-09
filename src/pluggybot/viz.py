"""What PluggyBot sees, in one PNG (issue #1).

A 2x2 dashboard image, refreshed on the same cadence as map.png:

    +-----------+-----------+
    | left_eye  | right_eye |     top row: the stereo pair (mapping eyes)
    +-----------+-----------+
    | occupancy | dock_eye  |     bottom: the map + the docking camera
    +-----------+-----------+

One offscreen Renderer serves all three cameras -- update_scene() just
switches which camera it looks through, so the cost is three renders plus a
map rescale per save. At the map.png cadence (2 Hz sim time) that is well
under the price the outlet spotter already pays for its own renders, which
is why the feature is flag-gated rather than always-on: scripts that never
ask for it never create the renderer at all.
"""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

TILE_W, TILE_H = 320, 180        # per camera tile; the composite is 640x360
BACKGROUND = 40                  # letterbox gray behind the map tile
SEAM = 2                         # px gap between tiles, drawn in BACKGROUND

CAMERAS = ("left_eye", "right_eye", "dock_eye")

BATTERY_OK = (80, 220, 100)      # > 50 %
BATTERY_LOW = (235, 180, 50)     # 20-50 %
BATTERY_CRITICAL = (230, 70, 60)  # < 20 %
BATTERY_CHARGING = (80, 180, 235)


class ViewDashboard:
  """Renders the composite view image. Reuse one instance per run.

  map_img is optional so camera-only scripts (teleop.py) can still use the
  dashboard: without it the map tile is a labeled blank.
  """

  def __init__(self, model, tile_w: int = TILE_W, tile_h: int = TILE_H) -> None:
    self.tile_w, self.tile_h = tile_w, tile_h
    self.renderer = mujoco.Renderer(model, tile_h, tile_w)

  def _camera_tile(self, data, name: str) -> np.ndarray:
    self.renderer.update_scene(data, camera=name)
    return self.renderer.render()

  def _map_tile(self, map_img: np.ndarray | None) -> np.ndarray:
    tile = np.full((self.tile_h, self.tile_w, 3), BACKGROUND, dtype=np.uint8)
    if map_img is None:
      return tile
    if map_img.ndim == 2:                       # grayscale -> RGB
      map_img = np.stack([map_img] * 3, axis=-1)
    # Fit preserving aspect; NEAREST keeps grid cells crisp instead of smearing
    # occupancy values into in-between grays (the renderer-as-measurement
    # lesson: this image is data, not scenery).
    scale = min(self.tile_w / map_img.shape[1], self.tile_h / map_img.shape[0])
    w, h = round(map_img.shape[1] * scale), round(map_img.shape[0] * scale)
    scaled = np.asarray(
      Image.fromarray(map_img).resize((w, h), Image.NEAREST))
    x0, y0 = (self.tile_w - w) // 2, (self.tile_h - h) // 2
    tile[y0:y0 + h, x0:x0 + w] = scaled
    return tile

  def _draw_battery(self, draw, x0: int, y0: int,
                    fraction: float, charging: bool) -> None:
    """Compact battery glyph: outlined bar + tip, fill scaled by charge and
    colored by urgency (blue while charging), with the percentage beside."""
    bw, bh = 40, 12
    color = (BATTERY_CHARGING if charging
             else BATTERY_OK if fraction > 0.5
             else BATTERY_LOW if fraction > 0.2
             else BATTERY_CRITICAL)
    draw.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(230, 230, 230))
    draw.rectangle([x0 + bw + 1, y0 + 3, x0 + bw + 4, y0 + bh - 3],
                   fill=(230, 230, 230))                    # the + terminal nub
    fill_w = max(1, round((bw - 4) * min(max(fraction, 0.0), 1.0)))
    draw.rectangle([x0 + 2, y0 + 2, x0 + 2 + fill_w, y0 + bh - 2], fill=color)
    label = f"{fraction:.0%}" + ("+" if charging else "")
    draw.text((x0 + bw + 10, y0 + 1), label, fill=color)

  def render(self, data, map_img: np.ndarray | None = None,
             battery: float | None = None, charging: bool = False) -> np.ndarray:
    """The composite frame as an (2*tile_h+SEAM, 2*tile_w+SEAM, 3) uint8 array.

    battery (0..1) draws a charge glyph in the map tile's top-right corner --
    the map tile, because that is where a human watches mission state (the
    map IMAGE itself stays untouched: map.png is a data surface where every
    pixel is an occupancy cell, so the HUD lives only on the dashboard).
    """
    left, right, dock = (self._camera_tile(data, c) for c in CAMERAS)
    tiles = ((left, "left eye"), (right, "right eye"),
             (self._map_tile(map_img), "map"), (dock, "dock eye"))

    th, tw = self.tile_h, self.tile_w
    out = np.full((2 * th + SEAM, 2 * tw + SEAM, 3), BACKGROUND, dtype=np.uint8)
    for i, (tile, label) in enumerate(tiles):
      y0, x0 = (i // 2) * (th + SEAM), (i % 2) * (tw + SEAM)
      out[y0:y0 + th, x0:x0 + tw] = tile
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    for i, (_, label) in enumerate(tiles):
      y0, x0 = (i // 2) * (th + SEAM), (i % 2) * (tw + SEAM)
      draw.text((x0 + 5, y0 + 3), label, fill=(255, 220, 60))
    if battery is not None:
      self._draw_battery(draw, tw - 90, th + SEAM + 4, battery, charging)
    return np.asarray(img)

  def save(self, data, map_img: np.ndarray | None = None,
           path: str = "views.png", battery: float | None = None,
           charging: bool = False) -> None:
    Image.fromarray(self.render(data, map_img, battery, charging)).save(path)

  def close(self) -> None:
    self.renderer.close()
