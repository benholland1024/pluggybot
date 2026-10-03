"""The legs' drivers, commanded as the robot commands them (issue #385).

A joint's driver is a GDS68 in MIT mode: every command carries its own
target and PD gains, so a walking policy and a routine that pushes with
torque (the scripted fold, a lie-down) take turns on the same twelve
drivers. In the sim each is a MuJoCo `dcmotor` in position mode
(`model.body_xml`'s "position" drive; `actuator.JointLimits.dcmotor` has the
algebra), the PD and the torque-speed envelope in C on every physics step,
and a command writes the gains it carries into the model:

  pd(target, stiffness, damping)   a target and its gains, held between
                                   commands -- a policy's
  torque(tau)                      this step's torque, through the envelope;
                                   re-sent every step, as any torque is
  limp()                           holding nothing: no stiffness, no damping

A torque is a target offset by tau over the stiffness it is carried on,
with the controller's damping set to cancel the motor's back-EMF (kd = -K):
the actuator then damps nothing, `implicitfast` has nothing to integrate
implicitly, and a torque step is the plain torque motor's to 1e-14 N*m.

⚠ The gains live in the MODEL, not the data: a recompile (a tool hung,
#168) or a restart builds them fresh from the XML's defaults until the
next command writes its own.
"""

import mujoco
import numpy as np

from pluggybot.legs.model import JOINT_NAMES

#: Where MuJoCo 3.10 keeps a `dcmotor`'s numbers in `actuator_gainprm`.
R, K, KP, KD, VMAX = 0, 1, 4, 6, 7
#: The controller's stiffness a torque rides on, V/rad. Any positive one
#: gives the same torque; a constant, because the model's own gains are
#: whatever the last command wrote (after `limp`, zero).
TORQUE_KP_V = 10.0


class Drivers:
  """One body's twelve leg drivers, in `model.JOINT_NAMES` order."""

  def __init__(self, model, data, prefix: str = ""):
    self.prefix = prefix
    self.rebind(model, data)
    #: Whether the last command was a torque or limp, whose gains a held
    #: target must not inherit.
    self.torqued = False

  def rebind(self, model, data) -> None:
    """The drivers on a world (and a recompiled one, #407), by name. ⚠ A
    recompile builds the model from its spec, so the gains a command wrote
    into it are the defaults again until the next command writes them."""
    prefix = self.prefix
    self.m, self.d = model, data
    self.act = np.array([model.actuator(f"{prefix}{n}").id for n in JOINT_NAMES])
    self.qadr = np.array([model.jnt_qposadr[model.joint(f"{prefix}{n}").id]
                          for n in JOINT_NAMES])
    if not (model.actuator_dyntype[self.act] == mujoco.mjtDyn.mjDYN_DCMOTOR).all():
      raise ValueError("these legs have no drivers to command (torque motors): "
                       "build the body with body_xml(drive='position')")
    g = model.actuator_gainprm[self.act]
    self._r, self._k = g[:, R].copy(), g[:, K].copy()

  def gains(self) -> tuple[np.ndarray, np.ndarray]:
    """The torque PD (stiffness, damping) the drivers run now."""
    g = self.m.actuator_gainprm[self.act]
    return g[:, KP] * self._k / self._r, (g[:, KD] + self._k) * self._k / self._r

  def pd(self, target: np.ndarray, stiffness, damping) -> None:
    g = self.m.actuator_gainprm
    g[self.act, KP] = stiffness * self._r / self._k
    g[self.act, KD] = damping * self._r / self._k - self._k
    self.d.ctrl[self.act] = target
    self.torqued = False

  def torque(self, tau: np.ndarray) -> None:
    g = self.m.actuator_gainprm
    g[self.act, KP] = TORQUE_KP_V
    g[self.act, KD] = -self._k
    self.d.ctrl[self.act] = (self.d.qpos[self.qadr]
                             + tau * self._r / (self._k * TORQUE_KP_V))
    self.torqued = True

  def limp(self) -> None:
    g = self.m.actuator_gainprm
    g[self.act, KP] = 0.0
    g[self.act, KD] = -self._k
    self.torqued = True

  def set_bus(self, volts: float) -> None:
    """The pack at `volts`: only the drivers' voltage clamp moves (K and R
    are the motor's own)."""
    self.m.actuator_gainprm[self.act, VMAX] = volts
