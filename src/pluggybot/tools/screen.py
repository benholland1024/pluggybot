"""The LCD module's display: expressive faces and a text/count readout (#13).

`module_lcd` has been a body with a dark 40 mm panel on it since the rack was
built -- the one tool in the world whose whole purpose is DISPLAYING things,
and the only one that had nothing to display. This module is that content.

Three rules, and the first is the one that makes it cheap:

  THE FACE IS DRAWN IN THE BROWSER. Layer 3 of the three-layer model
  (activity/base.py): rigid bodies in MuJoCo, discrete state in Python,
  everything organic on the website. What travels the wire is an ENUM plus an
  animation hint, so an expressive robot costs five short strings per change
  and no textures, no render targets, and no MuJoCo work at all. It is the
  same argument that keeps ink out of the model, applied to a face.

  THE SIM DOES NOT TICK ANIMATIONS. `hint` names a loop the browser runs
  ("blink", "bounce", "shake"); it is state, not an event. A blink is 150 ms
  of eyelid and the pose stream runs at 20 Hz, so animating one from here
  would cost frames to transmit a thing the browser can do for free -- and
  would look worse, because a dropped frame would stick the eyes shut.

  A DARK SCREEN IS THE HONEST DEFAULT. `powered` comes from the coupling's
  own electrical criterion (`module_power_contact`), not from "are we
  carrying it": a module hanging on the rack has no power, and a HALF-seated
  coupling -- one pole conducting -- has none either. So the face appears
  exactly when the robot has actually picked the LCD up and seated it, which
  is the same fact the tool-power draw is billed against.

The lifecycle drives the face by default (`face_for` below) and an errand's
use-phase may override it -- the census puts a number on the screen, the
dance choreographs expressions. The override holds until the mission's STATE
changes, at which point the automatic face takes back over; that is what
leaves a census result on the screen while the robot stands there, and
clears it when it drives away.
"""

from typing import Callable

from pluggybot.rack.coupling import module_power_contact
from pluggybot.telemetry.protocol import FACE_STATES, SCREEN_HINTS

#: Battery fraction below which the automatic face starts to look worried.
#: Not the lifecycle's charge threshold (that is absolute energy against the
#: worst return trip): this is a FEELING, and it may be wrong.
ANXIOUS_FRAC = 0.25


class Screen:
  """One display module's content, and whether it has the power to show it.

  The public surface is `flags` -- a small JSON-ready dict that goes to
  telemetry verbatim, on the same terms as an Activity's. Setting content
  while the module is unpowered is remembered but not shown: the mode reads
  `off` until the coupling conducts, and the moment it does the face is
  already there.
  """

  def __init__(self, model, data, module: str = "module_lcd",
               name: str | None = None) -> None:
    self.model, self.data = model, data
    self.module = module
    #: telemetry key. The MODULE's name, not the robot's: the display belongs
    #: to the device, which either robot may be carrying (or neither).
    self.name = name or module
    self.powered = False
    self.changes = 0
    self.on_change: list[Callable[[str, dict], None]] = []
    # What the screen WOULD show if it had power. Kept across a power cut so
    # a module put down mid-count still knows its number when picked back up.
    self._mode = "face"
    self._face = "idle"
    self._hint = "blink"
    self._text: str | None = None
    self._count: int | None = None
    self._label: str | None = None
    #: True while an errand has taken the screen over (see module docstring).
    self.held = False
    self.flags: dict = {}
    self._publish()

  # ---- what to show --------------------------------------------------------

  def face(self, face: str, hint: str = "blink", hold: bool = False) -> None:
    """Show an expressive face. `hold` claims the screen for an errand."""
    if face not in FACE_STATES:
      raise ValueError(f"unknown face {face!r} -- the vocabulary is "
                       f"{FACE_STATES} (adding one is a two-repo change)")
    if hint not in SCREEN_HINTS:
      raise ValueError(f"unknown hint {hint!r} -- the vocabulary is "
                       f"{SCREEN_HINTS}")
    self._mode, self._face, self._hint = "face", face, hint
    self.held = self.held or hold
    self._publish()

  def show_text(self, text: str, face: str = "happy",
                hint: str = "none") -> None:
    """Show a short string. Long strings are the browser's problem to wrap,
    but a screen is 45 mm wide, so keep it to a few characters."""
    self._mode, self._text = "text", str(text)
    self._face, self._hint = face, hint
    self.held = True
    self._publish()

  def show_count(self, count: int, label: str | None = None,
                 face: str = "happy", hint: str = "none") -> None:
    """Show a number and what it counts -- the census's whole output."""
    self._mode, self._count, self._label = "count", int(count), label
    self._face, self._hint = face, hint
    self.held = True
    self._publish()

  def blank(self) -> None:
    """Deliberately show nothing (distinct from having no power)."""
    self._mode = "off"
    self.held = True
    self._publish()

  def release(self) -> None:
    """Give the screen back to the automatic face."""
    self.held = False

  # ---- the physics seam ----------------------------------------------------

  def sense(self, model, data, powered: bool | None = None) -> None:
    """Read the coupling. Called every physics step; must stay cheap.

    `powered` lets a caller that has ALREADY asked the coupling this step
    hand the answer in rather than pay for a second O(ncon) scan -- the
    lifecycle does exactly that whenever the display is the errand's module.
    """
    if powered is None:
      powered = module_power_contact(model, data, self.module)
    if powered != self.powered:
      self.powered = powered
      self._publish()

  def step_hook(self) -> Callable[[], None]:
    """A zero-argument callback for `HubMission.step_hooks`."""
    def hook() -> None:
      self.sense(self.model, self.data)
    return hook

  # ---- telemetry -----------------------------------------------------------

  def _publish(self) -> None:
    """Recompute `flags`, firing `on_change` when anything actually moved.

    Sparse-emission bookkeeping deliberately does NOT live here -- it lives
    in the FrameBuilder, for the same reason an Activity's does: `serve.py
    --record` runs a publisher and a recorder over one world, and a shared
    "already sent" memory would have the two sinks eating each other's
    deltas.
    """
    mode = self._mode if self.powered else "off"
    flags: dict = {"mode": mode, "powered": self.powered}
    if mode != "off":
      flags["face"] = self._face
      flags["hint"] = self._hint
    if mode == "text":
      flags["text"] = self._text
    if mode == "count":
      flags["count"] = self._count
      flags["label"] = self._label
    if flags == self.flags:
      return
    self.flags = flags
    self.changes += 1
    for hook in self.on_change:
      hook(self.name, dict(flags))


class ScreenSet:
  """Every display in one world: one step hook, one snapshot.

  The same duck type an `ActivitySet` and a `BoardBook` present (`names` +
  `snapshot()`), because the FrameBuilder diffs all three with one code path.
  """

  def __init__(self, screens=()) -> None:
    self.screens: dict[str, Screen] = {s.name: s for s in screens}

  def __iter__(self):
    return iter(self.screens.values())

  def __len__(self) -> int:
    return len(self.screens)

  def __getitem__(self, name: str) -> Screen:
    return self.screens[name]

  def add(self, screen: Screen) -> Screen:
    self.screens[screen.name] = screen
    return screen

  def step_hook(self) -> Callable[[], None]:
    """A zero-argument callback for `HubMission.step_hooks`.

    For displays NOBODY ELSE drives. The hub lifecycle senses the screen it
    owns itself (`HubLifecycle._screen_step`), because it already has the
    coupling's answer in hand for that module and a second O(ncon) scan per
    physics step is not free -- wiring this as well would pay for the same
    question twice. A world with a display the lifecycle does not carry wants
    this hook; today's worlds have exactly one and it does.
    """
    def hook() -> None:
      for screen in self.screens.values():
        screen.sense(screen.model, screen.data)
    return hook

  def snapshot(self) -> dict:
    return {name: dict(s.flags) for name, s in self.screens.items()}

  @property
  def names(self) -> list[str]:
    return list(self.screens)


# ---- the automatic face ----------------------------------------------------

def face_for(state: str, battery_frac: float = 1.0,
             hunger: str = "") -> tuple[str, str]:
  """(face, hint) for a lifecycle state -- the robot's resting expression.

  Deliberately a pure function of state plus two readings, so it is testable
  without a physics world and so "what does the robot look like right now"
  has exactly one answer. An errand that wants something else says so.

  `hunger` is a `HUNGER_STATES` name or "" (issue #36), and it reaches the
  face through ONE door: `starving` replaces the `idle` default below. That
  narrowness is the point. A starving robot should show it, but every other
  branch here is about something more urgent than food -- a low pack, a
  charge in progress, a tool in the air -- and hunger stomping those would
  hide the states a watcher actually needs. It is also the whole of what
  being out of points does: the face changes and nothing else, which is
  issue #36's "zero is narrative, never a capability lock".
  """
  if state == "CHARGE":
    return "sleepy", "blink"
  if state == "GO_CHARGE":
    # Low enough to be interesting is low enough to be nervous about.
    return ("worried", "shake") if battery_frac < ANXIOUS_FRAC \
        else ("determined", "none")
  if state == "DECIDE":
    # Thinking about what to do next (issue #15). Deliberately distinct from
    # EXPLORE's curious/blink, and deliberately not a new vocabulary entry:
    # FACE_STATES and SCREEN_HINTS are a two-repo contract, so a new
    # expression is a pair of existing words before it is a new word.
    return "curious", "bounce"
  if state == "EXPLORE":
    return "curious", "blink"
  if state in ("SWAP_PICK", "SWAP_RETURN"):
    return "determined", "none"
  if state == "USE_TOOL":
    return "happy", "bounce"
  if state == "DONE":
    return "happy", "blink"
  if hunger == "starving":
    # Nothing else to be doing and nothing in the bank. Two existing words
    # rather than a new one: FACE_STATES is a two-repo contract, so a new
    # expression is a pair of old ones before it is a new entry.
    return "worried", "shake"
  return "idle", "blink"
