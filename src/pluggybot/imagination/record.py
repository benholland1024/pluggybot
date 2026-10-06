"""The record of a probe (issue #466): what the robot sent its arm and what
it sensed while it did -- the one thing the imagination's worker is given
besides a document. Plain numbers only:

  start     the robot as it knew itself when the record began: its believed
            pose in its map (x, y m, yaw rad), its tilt off its IMU (roll,
            pitch rad), its legs' and its arm's encoders (rad; the arm's as
            its drivers run them, the shoulder and the forearm's absolute
            angle), and the module on its fork
  commands  a row a physics step (`dt`): the arm drivers' targets (rad) and
            gains (Kp N*m/rad, Kd N*m*s/rad), the claw's slide (m) and jaws
            (m, each jaw's travel off shut, as `tools.claw` sends both) --
            what the robot SENT, so its own; it begins with nothing in the
            jaws, which a rollout puts where the first row sends them
  sensed    the same rows: the drivers' torque readings (N*m) and the
            encoders (rad), when the record is of a probe flown
  depth     point clouds in the map frame, m, laid there off the robot's
            belief at each frame: ⚠ THE POINTS ONLY. A depth
            frame carries the geom each pixel hit (`DepthFrame.peer_geoms`),
            the simulator's own label, and a record carries none (`cloud`)
  detections  the tags it decoded, numbers keyed by name
            (`TagDetector.detect_all`'s)

Its wire form (`to_wire`, `from_wire`) is a JSON header and named float
arrays, which is all `worker.py` will carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

#: The command columns, in order.
COMMANDS = ("shoulder", "fore", "kp", "kd", "slide", "jaws")
#: The sensed columns, in order: the two torque readings and the two encoders.
SENSED = ("torque_shoulder", "torque_elbow", "shoulder", "fore")


def _finite(a: np.ndarray, what: str) -> np.ndarray:
  a = np.asarray(a)
  if a.dtype.kind not in "fiu":
    raise ValueError(f"{what} must be numbers, not {a.dtype}")
  a = a.astype(float)
  if not np.isfinite(a).all():
    raise ValueError(f"{what} must be finite")
  return a


def cloud(points) -> np.ndarray:
  """A point cloud as a record carries it: (k, 3) metres and nothing else --
  ⚠ a column more (a geom id beside each point) is the simulator's label,
  and refused."""
  a = _finite(points, "a cloud")
  if a.ndim != 2 or a.shape[1] != 3:
    raise ValueError(f"a cloud is (k, 3) points, x y z; not {a.shape}")
  return a


def plain(v, what: str = "a detection"):
  """`v` as JSON-ready numbers and names: a tag detection's fields, nested."""
  if isinstance(v, bool) or v is None or isinstance(v, str):
    return v
  if isinstance(v, (int, np.integer)):
    return int(v)
  if isinstance(v, (float, np.floating)):
    if not math.isfinite(float(v)):
      raise ValueError(f"{what}: a number must be finite")
    return float(v)
  if isinstance(v, dict):
    if not all(isinstance(k, str) for k in v):
      raise ValueError(f"{what}: keys must be names")
    return {k: plain(x, what) for k, x in v.items()}
  if isinstance(v, (list, tuple, np.ndarray)):
    return [plain(x, what) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
  raise ValueError(f"{what}: {type(v).__name__} is not plain data")


@dataclass(frozen=True)
class Start:
  """The robot as it knew itself when the record began (the module
  docstring's `start`)."""
  pose: tuple[float, float, float]
  attitude: tuple[float, float]
  legs: tuple[float, ...]
  arm: tuple[float, float]
  carrying: str | None = None


@dataclass(frozen=True)
class Record:
  start: Start
  dt: float
  #: (n, 6): `COMMANDS`.
  commands: np.ndarray
  #: (n, 4): `SENSED`, or None for a path not yet flown.
  sensed: np.ndarray | None = None
  depth: tuple[np.ndarray, ...] = ()
  detections: tuple[dict, ...] = field(default=())

  def __post_init__(self) -> None:
    c = _finite(self.commands, "the commands")
    if c.ndim != 2 or c.shape[1] != len(COMMANDS) or len(c) == 0:
      raise ValueError(f"the commands are rows of {', '.join(COMMANDS)}")
    object.__setattr__(self, "commands", c)
    if self.sensed is not None:
      s = _finite(self.sensed, "what was sensed")
      if s.shape != (len(c), len(SENSED)):
        raise ValueError(f"what was sensed is a row of {', '.join(SENSED)} a command")
      object.__setattr__(self, "sensed", s)
    if not (isinstance(self.dt, (int, float)) and self.dt > 0):
      raise ValueError("dt is the rows' period, s")
    object.__setattr__(self, "depth", tuple(cloud(p) for p in self.depth))
    object.__setattr__(self, "detections", tuple(plain(t) for t in self.detections))
    st = self.start
    if len(st.pose) != 3 or len(st.attitude) != 2 or len(st.legs) != 12 or len(st.arm) != 2:
      raise ValueError("a start is a pose (x, y, yaw), an attitude (roll, pitch), "
                       "twelve legs' angles and the arm's two")

  @property
  def n(self) -> int:
    return len(self.commands)

  def column(self, name: str) -> np.ndarray:
    return self.commands[:, COMMANDS.index(name)]

  # ---- the wire ------------------------------------------------------------------

  def to_wire(self) -> tuple[dict, dict[str, np.ndarray]]:
    """(a JSON-ready header, named float arrays)."""
    st = self.start
    head = {"start": {"pose": [float(v) for v in st.pose],
                      "attitude": [float(v) for v in st.attitude],
                      "legs": [float(v) for v in st.legs], "arm": [float(v) for v in st.arm],
                      "carrying": st.carrying},
            "dt": float(self.dt), "depth": len(self.depth),
            "detections": list(self.detections)}
    arrays = {"commands": self.commands}
    if self.sensed is not None:
      arrays["sensed"] = self.sensed
    arrays.update({f"depth{k}": p for k, p in enumerate(self.depth)})
    return head, arrays

  @classmethod
  def from_wire(cls, head: dict, arrays: dict) -> "Record":
    s = head["start"]
    start = Start(pose=tuple(s["pose"]), attitude=tuple(s["attitude"]),
                  legs=tuple(s["legs"]), arm=tuple(s["arm"]), carrying=s["carrying"])
    return cls(start=start, dt=float(head["dt"]), commands=arrays["commands"],
               sensed=arrays.get("sensed"),
               depth=tuple(arrays[f"depth{k}"] for k in range(int(head["depth"]))),
               detections=tuple(head["detections"]))
