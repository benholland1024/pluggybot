"""The robots' cameras draw through ONE renderer (issue #713).

Each of a humanoid's two cameras decodes tags through a `TagDetector`
(`perception/tags.py`): the colour imager's (`HumanoidBody._detector`) and
the navigation camera's (`HumanoidPlaces._place_detector`). Each built a
`mujoco.Renderer` of its own, ~210 MiB under osmesa, four in the served
pair; now every detector of one model and picture size draws through one
(`tags._renderer_for`), closed with its last user. Pinned here, on a small
world of two robots:

  1. one renderer, and every camera's picture through it the same bit for
     bit as through a renderer of its own, drawn in any order;
  2. a body's close releases its cameras' use, a model's renderer outlives
     one body's close while another body draws through it, and nothing is
     kept once the last user is closed -- or collected unclosed;
  3. a renderer is closed with its own GL context current, so another's
     pictures are as they were (under osmesa they were not); a draw holds
     the lock; one collected unclosed is closed at the next draw, build or
     close, never inside the collection;
  4. a forked child builds its own.
"""

import gc
import math
import threading
import weakref

import mujoco
import numpy as np
import pytest

from pluggybot.body import body_for
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import world as hw
from pluggybot.perception import tags
from pluggybot.robot import FIRST, SECOND, SECOND_CHASSIS_RGBA

#: A robot's two cameras that decode tags.
CAMERAS = ("color_eye", "nav_eye")


def _stand(model, data, handle, x: float, y: float, yaw: float) -> None:
  q = handle.qpos_adr(model)
  data.qpos[q:q + 3] = [x, y, hm.stand_height()]
  data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  stand = hm.stand_qpos()
  for name in hm.qpos_order():
    data.qpos[model.jnt_qposadr[model.joint(handle.el(name)).id]] = stand.get(name, 0.0)


def _pair():
  """Two humanoids 2 m apart on a bare floor, facing each other, boxes of
  four colours round them: each camera sees the other robot and some."""
  props = "".join(
    f'<geom name="box{i}" type="box" size="0.2 0.2 {h / 2}" pos="{x} {y} {h / 2}" '
    f'rgba="{r} 0.5 {1 - r} 1"/>'
    for i, (x, y, h, r) in enumerate(((1.0, -0.8, 0.6, 0.1), (1.2, 0.9, 1.0, 0.4),
                                      (0.4, 1.5, 1.4, 0.7), (0.8, 0.1, 0.3, 0.9))))
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, scenery=props))
  hw._attach_humanoid(spec, (0.0, 0.0), SECOND.prefix, SECOND_CHASSIS_RGBA)
  model = spec.compile()
  data = mujoco.MjData(model)
  _stand(model, data, FIRST, 0.0, 0.0, 0.0)
  _stand(model, data, SECOND, 2.0, 0.3, math.pi)
  mujoco.mj_forward(model, data)
  return model, data


def _own(model, data, camera: str) -> np.ndarray:
  """The camera's picture through a renderer of its own, as every detector
  drew before #713: the detectors' scene flags, nothing drawn before it."""
  renderer = mujoco.Renderer(model, tags.PICTURE_H, tags.PICTURE_W)
  try:
    for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
      renderer.scene.flags[flag] = 0
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()
  finally:
    renderer.close()


def _renderers_of(model) -> list:
  """The shared renderers kept for `model` (the registry's own key)."""
  return [entry[0] for key, entry in tags._RENDERERS.items() if key[1] == id(model)]


def test_the_cameras_draw_through_one_renderer_each_picture_as_its_own():
  """Four renderers of one model at one size, drawn one at a time on the
  physics thread, did the work of one (~620 MiB of the served pair's ~1.5
  GB). Shared, a camera's picture must not depend on which camera drew
  before it: each is the same bit for bit as through a renderer of its
  own, in the order built, reversed and interleaved."""
  model, data = _pair()
  eyes = [tags.TagDetector(model, h.el(c)) for h in (FIRST, SECOND) for c in CAMERAS]
  try:
    assert len({id(e.renderer) for e in eyes}) == 1, "a renderer each"
    own = {e.camera_name: _own(model, data, e.camera_name) for e in eyes}
    assert len({p.tobytes() for p in own.values()}) == len(eyes), \
      "two cameras drew the same picture: nothing here tells them apart"
    for order in (eyes, eyes[::-1], eyes[0::2] + eyes[1::2]):
      for eye in order:
        assert np.array_equal(eye.frame(data), own[eye.camera_name]), \
          f"{eye.camera_name}'s picture differs drawn after another camera's"
  finally:
    for eye in eyes:
      eye.close()


def test_a_renderer_is_closed_with_its_last_user_and_nothing_is_kept():
  """Closing by a count of users: the tests' many bodies of many models
  would each keep a 210 MiB renderer alive in a worker otherwise. Bodies of
  two models, the pair's one model shared by two of them: the first robot's
  close leaves the second drawing; once every body is closed no renderer
  of either model is kept, nor one whose detector was dropped unclosed."""
  model, data = _pair()
  bare = mujoco.MjModel.from_xml_string(hm.body_xml())
  bare_data = mujoco.MjData(bare)
  mujoco.mj_resetDataKeyframe(bare, bare_data, 0)
  mujoco.mj_forward(bare, bare_data)
  first, second = body_for(model, data), body_for(model, data, handle=SECOND)
  alone = body_for(bare, bare_data)

  def draw(body):
    """Both its cameras, each building its detector at its first use."""
    body.detect_tags()
    body.mission.look_for_places()

  for body in (first, second, alone):
    draw(body)
  assert len(_renderers_of(model)) == 1 and len(_renderers_of(bare)) == 1
  kept = [weakref.ref(r) for r in _renderers_of(model) + _renderers_of(bare)]
  first.close()
  draw(second)
  assert len(_renderers_of(model)) == 1, "closed with its first user"
  second.close()
  alone.close()
  gc.collect()
  assert not _renderers_of(model) and not _renderers_of(bare)
  assert all(r() is None for r in kept), "a renderer kept after its users closed"

  eye = tags.TagDetector(bare, FIRST.el("nav_eye"))
  dropped = weakref.ref(eye.renderer)
  del eye
  gc.collect()
  tags.TagDetector(model, FIRST.el("nav_eye")).close()      # the next build
  assert dropped() is None and not _renderers_of(bare), \
    "a renderer kept for a detector collected unclosed"


class _Spy:
  """A renderer's GL context, the calls made on it written down."""

  def __init__(self, gl, calls: list):
    self.gl, self.calls = gl, calls

  def make_current(self):
    self.calls.append("current")
    self.gl.make_current()

  def free(self):
    self.calls.append("free")
    self.gl.free()


def test_a_renderer_is_closed_with_its_own_context_current_and_the_others_draw_on():
  """MuJoCo 3.10's `Renderer.close` frees its GL context and then its
  `MjrContext`, whose deletes go to whatever context is current: closed
  while another model's renderer was current, under osmesa that renderer's
  next picture came out wrong in 626,450 of its 921,600 pixels. Closed with
  its own context current, the other draws as it did (the pictures fail
  without it under osmesa alone; the order on any backend)."""
  model, data = _pair()
  other, other_data = _pair()
  closing = tags.TagDetector(model, FIRST.el("nav_eye"))
  kept = tags.TagDetector(other, FIRST.el("nav_eye"))
  try:
    closing.frame(data)
    before = kept.frame(other_data)
    calls = []
    closing.renderer._gl_context = _Spy(closing.renderer._gl_context, calls)
    closing.close()
    assert calls == ["current", "free"], "closed with another context current"
    assert np.array_equal(kept.frame(other_data), before), \
      "another renderer's picture changed when one was closed"
    with pytest.raises(RuntimeError, match="closed"):
      closing.frame(data)
  finally:
    kept.close()


def test_a_draw_holds_the_renderers_lock():
  """One scene and one context serve every camera of a model: a draw on
  another thread between this one's `update_scene` and its `render` would
  draw the other camera's picture here."""
  model, data = _pair()
  eye = tags.TagDetector(model, FIRST.el("nav_eye"))
  held = []
  render = eye.renderer.render

  def spied(*args, **kwargs):
    other = threading.Thread(target=lambda: held.append(
      not tags._RENDERERS_LOCK.acquire(blocking=False) or tags._RENDERERS_LOCK.release()))
    other.start()
    other.join()
    return render(*args, **kwargs)

  eye.renderer.render = spied
  try:
    eye.frame(data)
  finally:
    del eye.renderer.render
    eye.close()
  assert held == [True], "another thread could take the renderers mid-draw"


def test_a_renderer_collected_unclosed_is_closed_at_the_next_draw():
  """A cycle collection runs at any allocation and on any thread -- between
  another renderer's context made current and its draw (`np.empty` in
  `Renderer.render`) included -- and a renderer closed there leaves that
  draw with no context. One whose last user was collected unclosed, on this
  thread or another, waits for the next draw, build or close."""
  model, _data = _pair()
  other, other_data = _pair()
  drawing = tags.TagDetector(other, FIRST.el("nav_eye"))
  try:
    eye = tags.TagDetector(model, FIRST.el("nav_eye"))
    left = weakref.ref(eye.renderer)
    del eye
    gc.collect()
    assert left() is not None and left()._mjr_context is not None, \
      "closed inside the collection"
    drawing.frame(other_data)
    assert left() is None, "kept past the next draw"

    eyes = [tags.TagDetector(model, FIRST.el("nav_eye"))]
    left = weakref.ref(eyes[0].renderer)

    def collect():
      eyes.clear()
      gc.collect()

    collector = threading.Thread(target=collect)
    collector.start()
    collector.join()
    assert left() is not None and left()._mjr_context is not None, \
      "closed on the thread that collected its detector"
    drawing.frame(other_data)
    assert left() is None, "kept past the next draw"
  finally:
    drawing.close()


def test_a_forked_child_never_draws_through_its_parents_renderer(monkeypatch):
  """No GL context survives a fork, and the model a child inherits is its
  parent's object, id and all: the key carries the process (`_getpid`,
  faked here), so a detector built in a child builds its own renderer."""
  model, _data = _pair()
  parent = tags.TagDetector(model, FIRST.el("nav_eye"))
  try:
    pid = tags._getpid()
    monkeypatch.setattr(tags, "_getpid", lambda: pid + 1)
    child = tags.TagDetector(model, FIRST.el("color_eye"))
    monkeypatch.undo()
    try:
      assert child.renderer is not parent.renderer, "a child drew through its parent's"
    finally:
      child.close()
  finally:
    parent.close()
  assert not _renderers_of(model)


def test_a_close_that_fails_never_fails_a_draw_and_keeps_its_renderer(monkeypatch):
  """A pending renderer is closed at the next draw: one whose close raises
  is said and kept (its own `__del__` would close it later with whatever
  context is current), and the draw goes on."""
  model, _data = _pair()
  other, other_data = _pair()
  drawing = tags.TagDetector(other, FIRST.el("nav_eye"))
  eye = tags.TagDetector(model, FIRST.el("nav_eye"))
  failing = eye.renderer
  try:
    del eye
    gc.collect()

    def fails(renderer):
      raise RuntimeError("no context")

    monkeypatch.setattr(tags, "_close", fails)
    with pytest.warns(RuntimeWarning, match="not closed"):
      drawing.frame(other_data)
    monkeypatch.undo()
    assert failing in tags._UNCLOSED and failing not in tags._PENDING
  finally:
    monkeypatch.undo()
    if failing in tags._UNCLOSED:
      tags._UNCLOSED.remove(failing)
    tags._close(failing)
    drawing.close()
