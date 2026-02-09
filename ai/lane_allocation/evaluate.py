"""
Evaluation script for trained RL lane allocation model.
Runs the policy and reports traffic flow metrics.
"""

import os

import numpy as np
from stable_baselines3 import PPO

from ai.lane_allocation.environment import load_rl_env


def evaluate_model(
    config,
    model_path: str | None = None,
    num_episodes: int | None = None,
    render: bool = False,
):
    """
    Evaluate a trained PPO model on the highway environment.

    Args:
        config: RLConfig instance
        model_path: Path to the saved model (overrides config)
        num_episodes: Number of episodes to run (overrides config)
        render: Whether to render the environment (overrides config)
    """
    eval_config = config.get_evaluation_config()
    env_config = config.get_environment_config()
    train_config = config.get_training_config()

    # Use provided values or fall back to config
    model_path = model_path or eval_config.get(
        "model_path", "./runtime/logs/ppo_lane_final"
    )
    num_episodes = num_episodes or eval_config.get("num_episodes", 10)
    render = render or eval_config.get("render", False)

    # Get device from training config
    device = train_config.get("device", "cpu")

    print("=" * 60)
    print(f"Evaluating model: {model_path}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        return

    model = PPO.load(model_path, device=device)
    print("✓ Model loaded successfully")

    # Create environment with config
    env = load_rl_env(env_config)
    print("✓ Environment created")

    # Run evaluation episodes
    all_metrics = []
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}:")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if render and step_count % 10 == 0:
                env.render()

            done = done or truncated

        # Get episode metrics
        metrics = env.get_metrics()
        all_metrics.append(metrics)
        episode_rewards.append(total_reward)

        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(
            f"  Avg Speed: {metrics.get('avg_speed', 0):.2f} m/s ({metrics.get('avg_speed', 0) * 3.6:.1f} km/h)"
        )
        print(f"  Speed Std: {metrics.get('speed_std', 0):.2f}")
        print(f"  Lane Changes: {metrics.get('lane_changes', 0)}")
        print(f"  Hard Braking: {metrics.get('hard_braking_events', 0)}")
        print(f"  Collisions: {metrics.get('collisions', 0)}")

    # Aggregate statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    avg_speed = np.mean([m["avg_speed"] for m in all_metrics])
    avg_speed_std = np.mean([m["speed_std"] for m in all_metrics])
    avg_lane_changes = np.mean([m["lane_changes"] for m in all_metrics])
    avg_hard_braking = np.mean([m["hard_braking_events"] for m in all_metrics])
    total_collisions = sum([m["collisions"] for m in all_metrics])
    avg_reward = np.mean(episode_rewards)

    print(f"Average Speed: {avg_speed:.2f} m/s ({avg_speed * 3.6:.1f} km/h)")
    print(f"Average Speed Std: {avg_speed_std:.2f} (stop-and-go indicator)")
    print(f"Average Lane Changes per Episode: {avg_lane_changes:.1f}")
    print(f"Average Hard Braking Events: {avg_hard_braking:.1f}")
    print(f"Total Collisions: {total_collisions}")
    print(f"Average Episode Reward: {avg_reward:.2f}")

    # Performance rating
    print("\n" + "=" * 60)
    print("PERFORMANCE RATING")
    print("=" * 60)

    score = 0
    comments = []

    # Get evaluation thresholds from config
    thresholds = config.get("evaluation.thresholds", {})
    excellent_speed = thresholds.get("excellent_speed", 28.0)
    good_speed = thresholds.get("good_speed", 22.0)
    smooth_std = thresholds.get("smooth_std", 3.0)
    moderate_std = thresholds.get("moderate_std", 6.0)
    efficient_lane_changes = thresholds.get("efficient_lane_changes", 5)
    moderate_lane_changes = thresholds.get("moderate_lane_changes", 10)

    # Speed score
    if avg_speed > excellent_speed:
        score += 25
        comments.append("✓ Excellent speed maintenance")
    elif avg_speed > good_speed:
        score += 15
        comments.append("○ Good speed, could be faster")
    else:
        score += 5
        comments.append("✗ Low average speed")

    # Smoothness score
    if avg_speed_std < smooth_std:
        score += 25
        comments.append("✓ Very smooth traffic flow")
    elif avg_speed_std < moderate_std:
        score += 15
        comments.append("○ Moderate stop-and-go behavior")
    else:
        score += 5
        comments.append("✗ High stop-and-go (poor flow)")

    # Safety score
    if total_collisions == 0:
        score += 25
        comments.append("✓ No collisions (safe)")
    else:
        score += 0
        comments.append(f"✗ {total_collisions} collisions detected")

    # Efficiency score
    if avg_lane_changes < efficient_lane_changes:
        score += 25
        comments.append("✓ Efficient lane usage")
    elif avg_lane_changes < moderate_lane_changes:
        score += 15
        comments.append("○ Moderate lane changes")
    else:
        score += 5
        comments.append("✗ Excessive lane changes")

    print(f"Overall Score: {score}/100")
    print("\nBreakdown:")
    for comment in comments:
        print(f"  {comment}")

    if score >= 80:
        print("\n🏆 Excellent performance! Ready for deployment.")
    elif score >= 60:
        print("\n👍 Good performance. Consider fine-tuning.")
    else:
        print("\n⚠️  Poor performance. More training needed.")

    print("=" * 60)


if __name__ == "__main__":
    import glob
    from pathlib import Path

    from ai.lane_allocation.config_utils import create_evaluation_parser, load_config

    parser = create_evaluation_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.apply_cli_overrides(args)

    # Get model path - prioritize best_model.zip
    model_path = args.model if hasattr(args, "model") and args.model else None
    if model_path is None:
        # Try to find the best trained model
        log_dir = Path(config.get("training.log_dir", "./runtime/logs"))

        # Look for best_model.zip first (preferred)
        best_model = log_dir / "ppo_lane_allocation_*" / "best_model.zip"
        candidates = glob.glob(str(best_model))

        if candidates:
            # Get most recent best model
            model_path = str(
                Path(max(candidates, key=os.path.getmtime)).with_suffix("")
            )
            print(f"Using best model checkpoint: {model_path}")
        else:
            # Fall back to final model
            final_model = log_dir / "ppo_lane_allocation_*" / "ppo_lane_final.zip"
            candidates = glob.glob(str(final_model))

            if candidates:
                model_path = str(
                    Path(max(candidates, key=os.path.getmtime)).with_suffix("")
                )
                print(f"Using final model checkpoint: {model_path}")
            else:
                # Fall back to config default
                model_path = config.get(
                    "evaluation.model_path", "./runtime/logs/ppo_lane_final"
                )

        print(f"Auto-detected model: {model_path}")

    evaluate_model(
        config=config,
        model_path=model_path,
        num_episodes=(
            args.episodes if hasattr(args, "episodes") and args.episodes else None
        ),
        render=args.render if hasattr(args, "render") else False,
    )
