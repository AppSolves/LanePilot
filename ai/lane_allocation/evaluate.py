"""
Evaluation script for trained RL lane allocation model.
Runs the policy and reports traffic flow metrics.
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from ai.lane_allocation.environment import load_rl_env


def evaluate_model(model_path: str, num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained PPO model on the highway environment.

    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    print("=" * 60)
    print(f"Evaluating model: {model_path}")
    print("=" * 60)

    # Load model
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        return

    model = PPO.load(model_path)
    print("✓ Model loaded successfully")

    # Create environment
    config = {
        "num_lanes": 3,
        "road_length": 1000.0,
        "dt": 0.2,
        "spawn_rate": 0.8,  # High traffic density for testing
        "max_episode_steps": 300,
        "max_speed": 33.33,
        "min_speed": 8.33,
    }
    env = load_rl_env(config)
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

    # Speed score
    if avg_speed > 28:  # > 100 km/h
        score += 25
        comments.append("✓ Excellent speed maintenance")
    elif avg_speed > 22:  # > 80 km/h
        score += 15
        comments.append("○ Good speed, could be faster")
    else:
        score += 5
        comments.append("✗ Low average speed")

    # Smoothness score
    if avg_speed_std < 3:
        score += 25
        comments.append("✓ Very smooth traffic flow")
    elif avg_speed_std < 6:
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
    if avg_lane_changes < 5:
        score += 25
        comments.append("✓ Efficient lane usage")
    elif avg_lane_changes < 10:
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
    parser = argparse.ArgumentParser(description="Evaluate RL lane allocation model")
    parser.add_argument(
        "--model",
        type=str,
        default="./logs/ppo_lane_final",
        help="Path to trained model (without .zip extension)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )

    args = parser.parse_args()

    evaluate_model(args.model, args.episodes, args.render)
