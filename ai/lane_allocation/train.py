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
    import argparse

    parser = argparse.ArgumentParser(description="Train RL Lane Allocation Agent")
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--attention", action="store_true", default=True, help="Use attention policy"
    )
    parser.add_argument(
        "--mlp", action="store_true", help="Use MLP policy instead of attention"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument("--logdir", type=str, default="./logs", help="Log directory")
    parser.add_argument(
        "--spawn-rate", type=float, default=0.3, help="Initial spawn rate"
    )
    args = parser.parse_args()

    # Override attention if MLP specified
    use_attention = args.attention and not args.mlp

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(args.logdir, f"ppo_lane_allocation_{timestamp}")
    os.makedirs(logdir, exist_ok=True)

    print("=" * 60)
    print("Training RL Lane Allocation Agent")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Policy: {'Attention' if use_attention else 'MLP'}")
    print(f"Parallel Envs: {args.n_envs}")

    # Environment configuration
    config = {
        "num_lanes": 3,
        "road_length": 1000.0,
        "dt": 0.2,
        "spawn_rate": args.spawn_rate,  # Start with lower traffic, curriculum will increase
        "max_episode_steps": 300,
        "max_speed": 33.33,  # ~120 km/h
        "min_speed": 8.33,  # ~30 km/h
    }

    # Create environments
    def make_env():
        env = load_rl_env(config)
        env = Monitor(env)
        return env

    # Training environment
    vec_env = DummyVecEnv([make_env for _ in range(args.n_envs)])  # Parallel envs

    # Evaluation environment
    eval_env = DummyVecEnv([make_env])

    print(
        f"\nEnvironment: {config['num_lanes']} lanes, {config['road_length']}m length"
    )
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    policy_kwargs = {
        "features_extractor_class": (
            AttentionPolicyNetwork if use_attention else SimpleMLPExtractor
        ),
        "features_extractor_kwargs": (
            {
                "features_dim": 128,
                "num_attention_heads": 4,
                "hidden_dim": 128,
            }
            if use_attention
            else {"features_dim": 128}
        ),
        "net_arch": [
            dict(pi=[128, 64], vf=[128, 64])
        ],  # Separate networks for policy and value
    }

    print(f"\nUsing {'Attention-based' if use_attention else 'MLP'} policy network")

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=logdir,
        device=args.device,
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=logdir,
        name_prefix="ppo_lane",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=logdir,
        log_path=logdir,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    metrics_callback = MetricsCallback(verbose=1)
    curriculum_callback = CurriculumCallback(
        initial_spawn_rate=0.3,
        final_spawn_rate=0.8,
        steps_to_final=200_000,
        verbose=1,
    )

    callbacks = [
        checkpoint_callback,
        eval_callback,
        metrics_callback,
        curriculum_callback,
    ]

    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    # Train the model
    try:
        model.learn(
            total_timesteps=args.timesteps,
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
