"""PPO for the humanoid: mjlab's Go1 and G1 recipe (rsl_rl), unchanged but
for the experiment name -- the quadruped's before it (#377). The actor is
the network the served sim runs in numpy (`pluggybot.humanoid.policy`):
512-256-128, ELU, a running observation normaliser."""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class PpoWithSymmetryCfg(RslRlPpoAlgorithmCfg):
  """mjlab's PPO with rsl_rl's symmetry extension, which mjlab does not
  list (a task's own runner sets it: the CLI reads a dict's values as
  strings, and "False" is true)."""

  symmetry_cfg: dict | None = None


def ppo_runner_cfg(mirror: str | None = None) -> RslRlOnPolicyRunnerCfg:
  """`mirror`: a task's mirror image (`module:function`): rsl_rl's symmetry
  extension trains on each sample and its mirror (data augmentation)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
      # std_range is rsl_rl's default, listed so the CLI can cap the noise
      # (`--agent.actor.distribution-cfg.std-range 1e-6 0.15`).
      distribution_cfg={"class_name": "GaussianDistribution",
                        "init_std": 1.0, "std_type": "scalar",
                        "std_range": (1e-6, 1e6)}),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
      entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
      learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
      symmetry_cfg=None if mirror is None else {
        "data_augmentation_func": mirror, "use_data_augmentation": True,
        "use_mirror_loss": False, "mirror_loss_coeff": 0.0}),
    experiment_name="pluggy_humanoid",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
