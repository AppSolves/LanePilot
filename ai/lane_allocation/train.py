import os
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import load_rl_env
from .model import AttentionPolicyNetwork, SimpleMLPExtractor


class MetricsCallback(BaseCallback):
    """Custom callback for logging traffic flow metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []

    def _on_step(self) -> bool:
        # Check if episode is done
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Access unwrapped environment
                if hasattr(self.training_env, "envs"):
                    env = self.training_env.envs[idx]  # type: ignore
                    if hasattr(env, "get_metrics"):
                        metrics = env.get_metrics()
                        self.episode_metrics.append(metrics)

                        if self.verbose > 0 and len(self.episode_metrics) % 10 == 0:
                            recent_metrics = self.episode_metrics[-10:]
                            avg_speed = np.mean(
                                [m.get("avg_speed", 0) for m in recent_metrics]
                            )
                            avg_lane_changes = np.mean(
                                [m.get("lane_changes", 0) for m in recent_metrics]
                            )
                            print(f"\n=== Last 10 Episodes Metrics ===")
                            print(f"Avg Speed: {avg_speed:.2f} m/s")
                            print(f"Avg Lane Changes: {avg_lane_changes:.1f}")
                            print(
                                f"Avg Reward: {np.mean([m.get('total_reward', 0) for m in recent_metrics]):.2f}"
                            )

        return True


class CurriculumCallback(BaseCallback):
    """Gradually increase spawn rate (traffic density) during training."""

    def __init__(
        self,
        initial_spawn_rate=0.3,
        final_spawn_rate=0.8,
        steps_to_final=200_000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.initial_spawn_rate = initial_spawn_rate
        self.final_spawn_rate = final_spawn_rate
        self.steps_to_final = steps_to_final

    def _on_step(self) -> bool:
        # Linear curriculum
        progress = min(1.0, self.num_timesteps / self.steps_to_final)
        current_spawn_rate = (
            self.initial_spawn_rate
            + (self.final_spawn_rate - self.initial_spawn_rate) * progress
        )

        # Update all environments
        if hasattr(self.training_env, "envs"):
            for env in self.training_env.envs:  # type: ignore
                if hasattr(env, "spawn_rate"):
                    env.spawn_rate = current_spawn_rate

        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"Spawn rate updated to: {current_spawn_rate:.3f}")

        return True


if __name__ == "__main__":
    from ai.lane_allocation.config_utils import create_training_parser, load_config

    # Parse arguments
    parser = create_training_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.apply_cli_overrides(args)

    # Get configuration sections
    env_config = config.get_environment_config()
    train_config = config.get_training_config()
    policy_config = config.get_policy_config()
    curriculum_config = config.get("curriculum", {})
    callbacks_config = config.get("callbacks", {})

    # Determine policy type
    use_attention = policy_config.get("type", "attention") == "attention"

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(
        train_config.get("log_dir", "./runtime/logs"),
        f"ppo_lane_allocation_{timestamp}",
    )
    os.makedirs(logdir, exist_ok=True)

    print("=" * 60)
    print("Training RL Lane Allocation Agent")
    print("=" * 60)
    print(f"Device: {train_config.get('device', 'cpu')}")
    print(f"Timesteps: {train_config.get('timesteps', 500000):,}")
    print(f"Policy: {'Attention' if use_attention else 'MLP'}")
    print(f"Parallel Envs: {train_config.get('n_envs', 4)}")
    print(f"Config file: {config.config_path}")

    # Create environments
    def make_env():
        env = load_rl_env(env_config)
        env = Monitor(env)
        return env

    # Training environment
    n_envs = train_config.get("n_envs", 4)
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])  # Parallel envs

    # Evaluation environment
    eval_env = DummyVecEnv([make_env])

    print(
        f"\nEnvironment: {env_config.get('num_lanes', 3)} lanes, {env_config.get('road_length', 1000)}m length"
    )
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Policy configuration
    features_dim = policy_config.get("features_dim", 128)
    net_arch = policy_config.get("net_arch", {"pi": [128, 64], "vf": [128, 64]})

    if use_attention:
        attention_config = policy_config.get("attention", {})
        features_extractor_kwargs = {
            "features_dim": features_dim,
            "num_attention_heads": attention_config.get("num_heads", 4),
            "hidden_dim": attention_config.get("hidden_dim", 128),
        }
    else:
        features_extractor_kwargs = {"features_dim": features_dim}

    policy_kwargs = {
        "features_extractor_class": (
            AttentionPolicyNetwork if use_attention else SimpleMLPExtractor
        ),
        "features_extractor_kwargs": features_extractor_kwargs,
        "net_arch": dict(
            pi=net_arch.get("pi", [128, 64]), vf=net_arch.get("vf", [128, 64])
        ),
    }

    print(f"\nUsing {'Attention-based' if use_attention else 'MLP'} policy network")

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=train_config.get("verbose", 1),
        learning_rate=train_config.get("learning_rate", 3e-4),
        n_steps=train_config.get("n_steps", 2048),
        batch_size=train_config.get("batch_size", 64),
        n_epochs=train_config.get("n_epochs", 10),
        gamma=train_config.get("gamma", 0.99),
        gae_lambda=train_config.get("gae_lambda", 0.95),
        clip_range=train_config.get("clip_range", 0.2),
        ent_coef=train_config.get("ent_coef", 0.01),
        vf_coef=train_config.get("vf_coef", 0.5),
        max_grad_norm=train_config.get("max_grad_norm", 0.5),
        tensorboard_log=logdir if train_config.get("tensorboard_log", True) else None,
        device=train_config.get("device", "cpu"),
        seed=train_config.get("seed"),
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    if callbacks_config.get("checkpoint", {}).get("enabled", True):
        checkpoint_callback = CheckpointCallback(
            save_freq=callbacks_config.get("checkpoint", {}).get("save_freq", 20000),
            save_path=logdir,
            name_prefix="ppo_lane",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

    # Evaluation callback
    if callbacks_config.get("evaluation", {}).get("enabled", True):
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=logdir,
            log_path=logdir,
            eval_freq=callbacks_config.get("evaluation", {}).get("eval_freq", 10000),
            n_eval_episodes=callbacks_config.get("evaluation", {}).get(
                "n_eval_episodes", 5
            ),
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    # Metrics callback
    if callbacks_config.get("metrics", {}).get("enabled", True):
        metrics_callback = MetricsCallback(
            verbose=callbacks_config.get("metrics", {}).get("print_freq", 10)
        )
        callbacks.append(metrics_callback)

    # Curriculum callback
    if curriculum_config.get("enabled", True):
        curriculum_callback = CurriculumCallback(
            initial_spawn_rate=curriculum_config.get("initial_spawn_rate", 0.3),
            final_spawn_rate=curriculum_config.get("final_spawn_rate", 0.8),
            steps_to_final=curriculum_config.get("steps_to_final", 200000),
            verbose=1,
        )
        callbacks.append(curriculum_callback)

    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    # Train the model
    try:
        model.learn(
            total_timesteps=train_config.get("timesteps", 500000),
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

    # Save final model
    final_path = os.path.join(logdir, "ppo_lane_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Logs saved to: {logdir}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {logdir}")
