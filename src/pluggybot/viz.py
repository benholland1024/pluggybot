"""Demo video: `Recorder` streams an offscreen camera to a video file.

The same offscreen renderer the robot's cameras use, sampled on a fixed
SIM-time cadence and streamed to an encoder, so a demo can be shared without
screen-recording the viewer (`--record` on a demo script).
"""

import os
import subprocess
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The demo scripts already own a TRACKING camera (mjCAMERA_TRACKING locks the
# look-at point to a body and holds a fixed offset), which is what makes the
# filmstrips frame the tool rather than the room. A video is that same camera
# sampled on a timer instead of at four storyboard moments.

VIDEO_W, VIDEO_H = 1280, 720     # the offscreen buffer's full size (see the
                                 # <global offwidth/offheight> in the model)
GIF_W, GIF_FPS = 640, 15         # GIFs get downscaled; 720p GIF is absurd
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
  -- e.g. a camera on the robot itself. That view exists inside the sim and
  no screen capture of the viewer can reach it.

  `brighten` lifts each frame's pixels AFTER it is rendered (a gamma: 1 is
  none). ⚠ Never the scene's light: the robot's own cameras see by it, and
  brightened, a tag decoded 3 cm elsewhere (#507).
  """

  def __init__(self, model, path: str, *, track_body: str | None = None,
               fps: int = 30, speed: float = 1.0,
               distance: float = 1.0, azimuth: float = 135.0,
               elevation: float = -18.0, inset_camera: str | None = None,
               size: tuple[int, int] = (VIDEO_W, VIDEO_H),
               brighten: float = 1.0) -> None:
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
    if brighten <= 0:
      raise ValueError(f"brighten is a gamma, over 0: {brighten}")
    self.interval = speed / fps            # sim seconds between frames
    self.fps = fps
    self._next = 0.0
    self._label = ""
    self.frames_written = 0
    self._lift = (255.0 * (np.arange(256) / 255.0) ** (1.0 / brighten)
                  ).round().astype(np.uint8) if brighten != 1.0 else None

    self.rebind(model, None)
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
      self._make_inset(model)

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
    """Caption burned into subsequent frames -- the filmstrip's stage names;
    a line each, cut to fit the frame."""
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
    until it is time for a frame. ⚠ A clock set back (a scene's reset puts
    `data.time` at its start again) is sampled on from where it now is:
    held to the old next sample, a scene that starts over is not filmed.
    """
    if data.time < self._next - self.interval:
      self._next = data.time
    if data.time < self._next:
      return
    self._next = data.time + self.interval
    self._write(self._frame(data))

  def hold(self, data, seconds: float) -> None:
    """The clip's last frame shown `seconds` more: rendered once, now, and
    written again and again, so a caption set at the end is read. ⚠ Nothing
    is stepped: a hold that stepped the sim moved a demo's reported state."""
    frame = self._frame(data)
    for _ in range(round(seconds * self.fps)):
      self._write(frame)

  def _write(self, frame: np.ndarray) -> None:
    self.writer.append_data(frame)
    self.frames_written += 1

  def _frame(self, data) -> np.ndarray:
    """One frame of the scene as it is now: rendered, brightened, the inset
    and the captions on it."""
    self._apply_camera(EASE)
    self.renderer.update_scene(data, self.cam)
    img = Image.fromarray(self._lifted(self.renderer.render()))

    if self.inset is not None:
      self.inset.update_scene(data, camera=self.inset_camera)
      tile = Image.fromarray(self._lifted(self.inset.render()))
      m = self.height // 36
      x0, y0 = self.width - tile.width - m, self.height - tile.height - m
      img.paste(tile, (x0, y0))
      ImageDraw.Draw(img).rectangle(
        [x0, y0, x0 + tile.width - 1, y0 + tile.height - 1],
        outline=INSET_EDGE, width=2)

    self._caption(img, data.time)
    return np.asarray(img)

  def _lifted(self, pixels: np.ndarray) -> np.ndarray:
    """A rendered frame's pixels, brightened (`brighten`): a copy, the
    renderer's buffer left as it is."""
    return pixels.copy() if self._lift is None else self._lift[pixels]

  def _caption(self, img: Image.Image, sim_time: float) -> None:
    d = ImageDraw.Draw(img)
    m = self.height // 30
    clock = f"t = {sim_time:5.1f} s"
    texts = [(clock, "ra", (self.width - m, m)),
             (self.inset_camera or "", "rd", (self.width - m, self.height - m))]
    # The label's lines down the left, the first beside the clock: each cut
    # to the width it has, or it ran under the clock and off the frame.
    room = self.width - 3 * m - self._font.getlength(clock)
    for k, line in enumerate(self._label.splitlines()):
      texts.append((self._fit(line, room if k == 0 else self.width - 2 * m),
                    "la", (m, m + k * round(1.3 * self._font.size))))
    for text, anchor, xy in texts:
      if not text:
        continue
      d.text((xy[0] + 2, xy[1] + 2), text, font=self._font,
             fill=SHADOW, anchor=anchor)      # drop shadow: the scene behind
      d.text(xy, text, font=self._font, fill=LABEL, anchor=anchor)

  def _fit(self, text: str, width: float) -> str:
    """`text`, cut at a word to fit `width` px, an ellipsis where it was (a
    first word too wide alone cut where it reaches)."""
    if self._font.getlength(text) <= width:
      return text
    words = text.split(" ")
    while len(words) > 1 and self._font.getlength(" ".join(words) + " ...") > width:
      words.pop()
    cut = " ".join(words)
    while cut and self._font.getlength(cut + " ...") > width:
      cut = cut[:-1]
    return cut + " ..."


  def rebind(self, model, data) -> None:
    """A renderer is bound to its model."""
    self.model = model
    self.renderer = mujoco.Renderer(model, self.height, self.width)

  def _make_inset(self, model) -> None:
    iw = self.width // 4
    self.inset = mujoco.Renderer(model, (iw * self.height) // self.width, iw)
  def close(self, gif: str | None = None) -> str:
    """Finish the clip; with `gif`, a GIF of it there too (`GIF_W` wide, at
    `GIF_FPS` at most), made from the finished video."""
    self.writer.close()
    self.renderer.close()
    if self.inset is not None:
      self.inset.close()
    if gif is not None and gif != self.path:
      self._to_gif(gif)
    if self.is_gif:
      self._to_gif(self.path)
      os.unlink(self._mp4)
    return self.path

  def _to_gif(self, path: str) -> None:
    """mp4 -> GIF via a single shared palette (palettegen/paletteuse)."""
    import imageio_ffmpeg

    vf = (f"fps={min(self.fps, GIF_FPS)},scale={GIF_W}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=3")
    subprocess.run(
      [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
       "-i", self._mp4, "-vf", vf, "-loop", "0", path],
      check=True)
