"""
Interactive visualization tool for RL lane allocation model.
Displays highway simulation with real-time metrics and control.
"""

import os
import sys
import time
from pathlib import Path

import cv2
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.lane_allocation.config_utils import create_visualization_parser, load_config
from ai.lane_allocation.environment import load_rl_env


class VisualizationController:
    """Controls the visualization loop with keyboard interaction."""

    def __init__(self, config, model_path: str, lane_diversity: bool = False):
        """
        Initialize the visualization controller.

        Args:
            config: RLConfig instance
            model_path: Path to trained model
            lane_diversity: Cycle through multiple lane configurations [3,4,5,7]
        """
        self.config = config
        self.model_path = model_path

        # Get configurations
        self.env_config = config.get_environment_config()
        self.vis_config = config.get_visualization_config()
        self.eval_config = config.get_evaluation_config()
        self.train_config = config.get_training_config()

        # Set render mode for environment
        self.env_config["render_mode"] = "human"
        self.env_config["visualization"] = self.vis_config

        # Control settings
        controls = self.vis_config.get("controls", {})
        self.key_pause = controls.get("pause_key", 32)  # Space
        self.key_reset = controls.get("reset_key", ord("r"))
        self.key_quit = controls.get("quit_key", ord("q"))
        self.key_faster = controls.get("faster_key", ord("+"))
        self.key_slower = controls.get("slower_key", ord("-"))
        self.key_toggle_obs = controls.get("toggle_obs_key", ord("o"))

        # State
        self.paused = False
        self.running = True
        self.speed_multiplier = 1.0
        self.show_obs = False

        # Lane diversity
        self.lane_diversity = lane_diversity
        self.lane_configs = (
            [3, 4, 5, 7] if lane_diversity else [self.env_config.get("num_lanes", 3)]
        )
        self.initial_num_lanes = self.env_config.get("num_lanes", 3)
        if lane_diversity:
            print(
                f"🔄 Lane Diversity Mode: Will cycle through {self.lane_configs} lanes"
            )

        # Video recording
        self.recording_config = self.vis_config.get("recording", {})
        self.recording_enabled = self.recording_config.get("enabled", False)
        self.video_writer = None

        # Load model
        print("=" * 60)
        print("RL Lane Allocation Visualization")
        print("=" * 60)
        print(f"Loading model from: {model_path}")

        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Error: Model not found at {model_path}.zip")
            print("Train a model first:")
            print("  python -m ai.lane_allocation.train")
            sys.exit(1)

        # Load device from config
        device = self.train_config.get("device", "cpu")
        self.model = PPO.load(model_path, device=device)
        print(f"✓ Model loaded successfully (device: {device})")

        # Create environment
        print(f"✓ Environment: {self.env_config.get('num_lanes', 3)} lanes")
        print(f"✓ Spawn rate: {self.env_config.get('spawn_rate', 0.5)}")
        self.env = load_rl_env(self.env_config)

        # Initialize video recording if enabled
        if self.recording_enabled:
            self._init_video_recording()

        print("\n" + "=" * 60)
        print("CONTROLS:")
        print("  [SPACE]  Pause/Resume")
        print("  [R]      Reset episode")
        print("  [Q]      Quit")
        print("  [+]      Speed up simulation")
        print("  [-]      Slow down simulation")
        print("  [O]      Toggle observation overlay")
        print("=" * 60)
        print("\nStarting visualization...")
        print("Press any key in the visualization window to begin.\n")

    def _init_video_recording(self):
        """Initialize video recording."""
        output_dir = Path(self.recording_config.get("output_path", "./videos"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"lane_allocation_{timestamp}.mp4"

        # Video settings
        fps = self.recording_config.get("fps", 30)
        width = self.vis_config.get("window_width", 1400)
        height = self.vis_config.get("window_height", 900)

        # Create video writer
        fourcc = cv2.VideoWriter.fourcc(*self.recording_config.get("codec", "mp4v"))
        self.video_writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            fps,
            (width, height),
        )

        print(f"✓ Recording to: {output_file}")

    def run(self, num_episodes: int = 1):
        """
        Run the visualization loop.

        Args:
            num_episodes: Number of episodes to run (0 = infinite)
        """
        episode = 0

        try:
            while self.running and (num_episodes == 0 or episode < num_episodes):
                episode += 1
                # Cycle through lane configurations if lane diversity enabled
                if self.lane_diversity:
                    current_lanes = self.lane_configs[
                        (episode - 1) % len(self.lane_configs)
                    ]
                    self.env.num_lanes = current_lanes
                else:
                    current_lanes = self.lane_configs[0]
                self._run_episode(episode, current_lanes)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        finally:
            self._cleanup()

    def _run_episode(self, episode_num: int, num_lanes: int):
        """Run a single episode."""
        print(f"\n{'='*60}")
        if self.lane_diversity:
            print(f"Episode {episode_num} - Testing with {num_lanes} lanes")
        else:
            print(f"Episode {episode_num}")
        print(f"{'='*60}")

        obs, _ = self.env.reset()
        done = False
        step = 0
        episode_reward = 0

        start_time = time.time()

        while not done and self.running:
            # Handle pause
            while self.paused and self.running:
                key = cv2.waitKey(100) & 0xFF
                if key == self.key_pause or key == ord(" "):
                    self.paused = False
                    print("▶️  Resumed")
                elif key == self.key_quit or key == ord("q") or key == ord("Q"):
                    self.running = False
                    print("👋 Quitting from pause...")
                    return
                elif key == self.key_reset or key == ord("r") or key == ord("R"):
                    print("🔄 Reset requested")
                    return

            if not self.running:
                break

            # Predict action
            action, _ = self.model.predict(obs, deterministic=True)

            # Take step
            obs, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            step += 1

            # Render
            frame = self.env.render()

            # Record frame if enabled
            if (
                self.recording_enabled
                and self.video_writer is not None
                and frame is not None
            ):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame_bgr)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                # _handle_key returns False if should quit
                self.running = False
                break

            # Control simulation speed
            if self.speed_multiplier < 1.0:
                time.sleep(0.01 * (1.0 / max(self.speed_multiplier, 0.1)))

            done = done or truncated

        # Episode summary
        elapsed = time.time() - start_time
        metrics = self.env.get_metrics()

        print(f"\nEpisode {episode_num} Complete:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(
            f"  Avg Speed: {metrics.get('avg_speed', 0):.2f} m/s ({metrics.get('avg_speed', 0) * 3.6:.1f} km/h)"
        )
        print(f"  Speed Std: {metrics.get('speed_std', 0):.2f}")
        print(f"  Lane Changes: {metrics.get('lane_changes', 0)}")
        print(f"  Hard Braking Events: {metrics.get('hard_braking_events', 0)}")
        print(f"  Collisions: {metrics.get('collisions', 0)}")

    def _handle_key(self, key: int) -> bool:
        """Handle keyboard input.

        Returns:
            bool: True to continue, False to quit
        """
        if key == -1 or key == 255:
            return True

        if key == self.key_pause or key == ord(" "):
            self.paused = not self.paused
            if self.paused:
                print("⏸️  Paused")
            else:
                print("▶️  Resumed")

        elif key == self.key_reset or key == ord("r") or key == ord("R"):
            print("🔄 Resetting episode...")
            # Will be handled by returning from _run_episode

        elif key == self.key_quit or key == ord("q") or key == ord("Q"):
            print("👋 Quitting...")
            self.running = False
            return False

        elif key == self.key_faster or key == ord("+") or key == ord("="):
            self.speed_multiplier = min(10.0, self.speed_multiplier * 1.5)
            print(f"⏩ Speed: {self.speed_multiplier:.1f}x")

        elif key == self.key_slower or key == ord("-") or key == ord("_"):
            self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
            print(f"⏪ Speed: {self.speed_multiplier:.1f}x")

        elif key == self.key_toggle_obs or key == ord("o") or key == ord("O"):
            self.show_obs = not self.show_obs
            self.env.render_config["observation"]["show_overlay"] = self.show_obs
            if self.env.renderer is not None:
                self.env.renderer.show_observation = self.show_obs
            print(f"👁️  Observation overlay: {'ON' if self.show_obs else 'OFF'}")

        return True

    def _cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        if self.video_writer is not None:
            self.video_writer.release()
            print("✓ Video saved")

        self.env.close()
        cv2.destroyAllWindows()
        print("✓ Visualization closed")


def main():
    """Main entry point."""
    parser = create_visualization_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.apply_cli_overrides(args)

    # Get model path
    model_path = args.model if hasattr(args, "model") and args.model else None
    if model_path is None:
        # Try to find the latest trained model
        log_dir = Path(config.get("training.log_dir", "./runtime/logs"))

        # Look for best_model.zip first (preferred)
        import glob

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
                # Fall back to default
                model_path = config.get(
                    "evaluation.model_path", "./runtime/logs/ppo_lane_final"
                )

        print(f"Auto-detected model: {model_path}")

    # Get number of episodes
    num_episodes = (
        args.episodes if hasattr(args, "episodes") and args.episodes else 0  # infinite
    )

    # Check lane diversity
    lane_diversity = (
        args.lane_diversity
        if hasattr(args, "lane_diversity") and args.lane_diversity
        else False
    )

    # Create and run controller
    controller = VisualizationController(config, model_path, lane_diversity)
    controller.run(num_episodes)


if __name__ == "__main__":
    main()
