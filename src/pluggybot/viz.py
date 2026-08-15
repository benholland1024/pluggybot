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

`Recorder` (below) is the moving-picture version of the same idea: the same
offscreen renderer, sampled on a fixed SIM-time cadence and streamed to a
video file, so a demo can be shared without screen-recording the viewer.
"""

import os
import subprocess
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


# --- video capture -------------------------------------------------------
# The demo scripts already own a TRACKING camera (mjCAMERA_TRACKING locks the
# look-at point to a body and holds a fixed offset), which is what makes the
# filmstrips frame the tool rather than the room. A video is that same camera
# sampled on a timer instead of at four storyboard moments.

VIDEO_W, VIDEO_H = 1280, 720     # the offscreen buffer's full size (see the
                                 # <global offwidth/offheight> in the model)
GIF_W = 640                      # GIFs get downscaled; 720p GIF is absurd
LABEL = (245, 246, 250)
SHADOW = (12, 13, 15)
INSET_EDGE = (232, 234, 238)
EASE = 0.10                      # per frame, toward the target camera pose;
                                 # ~1.2 s to settle at 30 fps


class Recorder:
  """Streams offscreen frames straight to an .mp4 or .gif.

  Frames are written to the encoder as they are rendered rather than
  accumulated: at 30 fps a 90 s demo is ~2700 frames, and 2700 x 1280 x 720 x 3
  bytes is ~7 GB of RAM if you hold them.

  The sampling clock is SIM time, not wall time, which is the whole advantage
  over a screen recording -- playback speed is exact and constant no matter
  what the machine was doing, and `speed` buys slow motion (0.25 = quarter
  speed) or a timelapse (4.0) for free by changing only the sample interval.

  inset_camera renders a second, model-mounted camera as a picture-in-picture
  -- e.g. `claw_eye`, the camera on the claw module itself. That view exists
  inside the sim and no screen capture of the viewer can reach it.
  """

  def __init__(self, model, path: str, *, track_body: str | None = None,
               fps: int = 30, speed: float = 1.0,
               distance: float = 1.0, azimuth: float = 135.0,
               elevation: float = -18.0, inset_camera: str | None = None,
               size: tuple[int, int] = (VIDEO_W, VIDEO_H)) -> None:
    try:
      import imageio.v2 as imageio
    except ModuleNotFoundError as exc:      # pragma: no cover - install hint
      raise SystemExit(
        "--record needs imageio: uv add imageio imageio-ffmpeg") from exc

    self.path = path
    self.is_gif = path.lower().endswith(".gif")
    # h264 wants both dimensions on a 16-px macroblock grid. Left alone,
    # imageio-ffmpeg RESAMPLES the frames to get there (180 -> 192), which
    # quietly resizes pixels we just rendered at exactly the right size.
    # Render on the grid instead and nothing is resampled. Floor, never
    # ceil: growing would risk overrunning the model's offscreen buffer.
    self.width, self.height = (max(16, (v // 16) * 16) for v in size)
    self.interval = speed / fps            # sim seconds between frames
    self.fps = fps
    self._next = 0.0
    self._label = ""
    self.frames_written = 0

    self.model = model
    self.renderer = mujoco.Renderer(model, self.height, self.width)
    self.cam = mujoco.MjvCamera()
    if track_body is not None:
      self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      self.cam.trackbodyid = model.body(track_body).id
    self.cam.distance, self.cam.azimuth = distance, azimuth
    self.cam.elevation = elevation
    self._target = {"azimuth": azimuth, "elevation": elevation,
                    "distance": distance}

    self.inset_camera = inset_camera
    self.inset = None
    if inset_camera is not None:
      iw = self.width // 4
      self.inset = mujoco.Renderer(model, (iw * self.height) // self.width, iw)

    self._font = self._load_font(max(14, self.height // 30))
    # A GIF is encoded from a finished video, not frame by frame: ffmpeg's
    # palettegen/paletteuse needs to see the whole clip to pick one 256-colour
    # palette, and that is the difference between a clean GIF and a dithered
    # mess several times the size.
    self._mp4 = (tempfile.mktemp(suffix=".mp4") if self.is_gif else path)
    self.writer = imageio.get_writer(
      self._mp4, fps=fps, quality=8, macro_block_size=16)

  @staticmethod
  def _load_font(size: int):
    try:
      return ImageFont.load_default(size=size)   # Pillow >= 10.1 scales this
    except Exception:                             # pragma: no cover
      return ImageFont.load_default()

  def set_label(self, text: str) -> None:
    """Caption burned into subsequent frames -- the filmstrip's stage names."""
    self._label = text

  def set_camera(self, *, azimuth: float | None = None,
                 elevation: float | None = None, distance: float | None = None,
                 track_body: str | None = None, cut: bool = False) -> None:
    """Aim the camera somewhere else, easing there over ~a second.

    One fixed azimuth cannot cover a whole demo: the angle that sees the pen
    against the board (az 60) puts a wall through the lens back at the rack,
    where az 150 is right. So the camera moves with the story. The move is
    interpolated rather than cut because a hard cut on a tracking camera reads
    as a glitch; pass cut=True when you do want the jump.

    Changing track_body always jumps -- the look-at point IS the body.
    """
    if track_body is not None:
      self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      self.cam.trackbodyid = self.model.body(track_body).id
    for key, value in (("azimuth", azimuth), ("elevation", elevation),
                       ("distance", distance)):
      if value is not None:
        self._target[key] = value
    if cut:
      self._apply_camera(1.0)

  def _apply_camera(self, ease: float) -> None:
    """Step the live camera a fraction of the way to the target pose."""
    # Azimuth wraps: 350 -> 10 is +20 degrees, not -340, so interpolate along
    # the shortest arc or the camera swings the long way round the scene.
    delta = (self._target["azimuth"] - self.cam.azimuth + 180.0) % 360.0 - 180.0
    self.cam.azimuth += delta * ease
    self.cam.elevation += (self._target["elevation"] - self.cam.elevation) * ease
    self.cam.distance += (self._target["distance"] - self.cam.distance) * ease

  def maybe_grab(self, data) -> None:
    """Render one frame if the sim clock has reached the next sample time.

    Safe to call from an on_step hook on every step; it is a clock comparison
    until it is time for a frame.
    """
    if data.time < self._next:
      return
    self._next = data.time + self.interval
    self._apply_camera(EASE)
    self.renderer.update_scene(data, self.cam)
    img = Image.fromarray(self.renderer.render().copy())

    if self.inset is not None:
      self.inset.update_scene(data, camera=self.inset_camera)
      tile = Image.fromarray(self.inset.render().copy())
      m = self.height // 36
      x0, y0 = self.width - tile.width - m, self.height - tile.height - m
      img.paste(tile, (x0, y0))
      ImageDraw.Draw(img).rectangle(
        [x0, y0, x0 + tile.width - 1, y0 + tile.height - 1],
        outline=INSET_EDGE, width=2)

    self._caption(img, data.time)
    self.writer.append_data(np.asarray(img))
    self.frames_written += 1

  def _caption(self, img: Image.Image, sim_time: float) -> None:
    d = ImageDraw.Draw(img)
    m = self.height // 30
    for text, anchor, xy in (
        (self._label, "la", (m, m)),
        (f"t = {sim_time:5.1f} s", "ra", (self.width - m, m)),
        (self.inset_camera or "", "rd", (self.width - m, self.height - m)),
    ):
      if not text:
        continue
      d.text((xy[0] + 2, xy[1] + 2), text, font=self._font,
             fill=SHADOW, anchor=anchor)      # drop shadow: the scene behind
      d.text(xy, text, font=self._font, fill=LABEL, anchor=anchor)

  def close(self) -> str:
    self.writer.close()
    self.renderer.close()
    if self.inset is not None:
      self.inset.close()
    if self.is_gif:
      self._to_gif()
      os.unlink(self._mp4)
    return self.path

  def _to_gif(self) -> None:
    """mp4 -> GIF via a single shared palette (palettegen/paletteuse)."""
    import imageio_ffmpeg

    vf = (f"fps={self.fps},scale={GIF_W}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=3")
    subprocess.run(
      [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
       "-i", self._mp4, "-vf", vf, "-loop", "0", self.path],
      check=True)
