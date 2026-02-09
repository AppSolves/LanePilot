"""
Configuration utilities for RL lane allocation.
Handles loading YAML config and CLI overrides.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


class RLConfig:
    """Configuration manager for RL training and inference."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Example:
            config.get("environment.num_lanes")
            config.get("training.learning_rate")

        Args:
            key: Configuration key in dot notation
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_environment_config(self) -> dict[str, Any]:
        """Get environment configuration."""
        return self.config.get("environment", {})

    def get_training_config(self) -> dict[str, Any]:
        """Get training configuration."""
        return self.config.get("training", {})

    def get_policy_config(self) -> dict[str, Any]:
        """Get policy network configuration."""
        return self.config.get("policy", {})

    def get_visualization_config(self) -> dict[str, Any]:
        """Get visualization configuration."""
        return self.config.get("visualization", {})

    def get_evaluation_config(self) -> dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get("evaluation", {})

    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """
        Apply CLI argument overrides to configuration.

        Args:
            args: Parsed command-line arguments
        """
        # Environment overrides
        if hasattr(args, "num_lanes") and args.num_lanes is not None:
            self.set("environment.num_lanes", args.num_lanes)
        if hasattr(args, "spawn_rate") and args.spawn_rate is not None:
            self.set("environment.spawn_rate", args.spawn_rate)
        if hasattr(args, "max_speed") and args.max_speed is not None:
            self.set("environment.max_speed", args.max_speed)
        if hasattr(args, "max_episode_steps") and args.max_episode_steps is not None:
            self.set("environment.max_episode_steps", args.max_episode_steps)

        # Training overrides
        if hasattr(args, "timesteps") and args.timesteps is not None:
            self.set("training.timesteps", args.timesteps)
        if hasattr(args, "device") and args.device is not None:
            self.set("training.device", args.device)
        if hasattr(args, "n_envs") and args.n_envs is not None:
            self.set("training.n_envs", args.n_envs)
        if hasattr(args, "learning_rate") and args.learning_rate is not None:
            self.set("training.learning_rate", args.learning_rate)
        if hasattr(args, "batch_size") and args.batch_size is not None:
            self.set("training.batch_size", args.batch_size)
        if hasattr(args, "seed") and args.seed is not None:
            self.set("training.seed", args.seed)

        # Policy overrides
        if hasattr(args, "policy_type") and args.policy_type is not None:
            self.set("policy.type", args.policy_type)
        elif hasattr(args, "mlp") and args.mlp:
            self.set("policy.type", "mlp")
        elif hasattr(args, "attention") and args.attention:
            self.set("policy.type", "attention")

        # Curriculum overrides
        if hasattr(args, "no_curriculum") and args.no_curriculum:
            self.set("curriculum.enabled", False)
        if hasattr(args, "initial_spawn_rate") and args.initial_spawn_rate is not None:
            self.set("curriculum.initial_spawn_rate", args.initial_spawn_rate)
        if hasattr(args, "final_spawn_rate") and args.final_spawn_rate is not None:
            self.set("curriculum.final_spawn_rate", args.final_spawn_rate)

        # Evaluation overrides
        if hasattr(args, "model") and args.model is not None:
            self.set("evaluation.model_path", args.model)
        if hasattr(args, "episodes") and args.episodes is not None:
            self.set("evaluation.num_episodes", args.episodes)
        if hasattr(args, "render") and args.render:
            self.set("evaluation.render", True)

        # Visualization overrides
        if hasattr(args, "fps") and args.fps is not None:
            self.set("visualization.fps", args.fps)
        if hasattr(args, "record") and args.record:
            self.set("visualization.recording.enabled", True)
        if hasattr(args, "width") and args.width is not None:
            self.set("visualization.window_width", args.width)
        if hasattr(args, "height") and args.height is not None:
            self.set("visualization.window_height", args.height)

        # Logging overrides
        if hasattr(args, "logdir") and args.logdir is not None:
            self.set("training.log_dir", args.logdir)
        if hasattr(args, "verbose") and args.verbose is not None:
            self.set("training.verbose", args.verbose)

    def __repr__(self) -> str:
        return f"RLConfig(config_path={self.config_path})"


def create_training_parser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(
        description="Train RL Lane Allocation Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ai/lane_allocation/config.yaml)",
    )

    # Environment
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--num-lanes", type=int, help="Number of lanes")
    env_group.add_argument("--spawn-rate", type=float, help="Vehicle spawn rate")
    env_group.add_argument("--max-speed", type=float, help="Maximum speed (m/s)")
    env_group.add_argument(
        "--max-episode-steps", type=int, help="Max steps per episode"
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--timesteps", type=int, help="Total training timesteps")
    train_group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Training device",
    )
    train_group.add_argument(
        "--n-envs", type=int, help="Number of parallel environments"
    )
    train_group.add_argument("--learning-rate", type=float, help="Learning rate")
    train_group.add_argument("--batch-size", type=int, help="Batch size")
    train_group.add_argument("--seed", type=int, help="Random seed")
    train_group.add_argument("--logdir", type=str, help="Log directory")
    train_group.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        help="Verbosity level",
    )

    # Policy
    policy_group = parser.add_argument_group("Policy")
    policy_group.add_argument(
        "--policy-type",
        type=str,
        choices=["attention", "mlp"],
        help="Policy network type",
    )
    policy_group.add_argument(
        "--attention",
        action="store_true",
        help="Use attention policy (default)",
    )
    policy_group.add_argument(
        "--mlp",
        action="store_true",
        help="Use MLP policy instead of attention",
    )

    # Curriculum
    curriculum_group = parser.add_argument_group("Curriculum Learning")
    curriculum_group.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning",
    )
    curriculum_group.add_argument(
        "--initial-spawn-rate",
        type=float,
        help="Initial spawn rate for curriculum",
    )
    curriculum_group.add_argument(
        "--final-spawn-rate",
        type=float,
        help="Final spawn rate for curriculum",
    )

    return parser


def create_evaluation_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate RL lane allocation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (without .zip extension)",
    )

    # Evaluation
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )

    # Environment overrides
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--spawn-rate", type=float, help="Vehicle spawn rate")
    env_group.add_argument("--num-lanes", type=int, help="Number of lanes")
    env_group.add_argument(
        "--lane-diversity",
        action="store_true",
        help="Test model on multiple lane configurations [3,4,5,7]",
    )

    return parser


def create_visualization_parser() -> argparse.ArgumentParser:
    """Create argument parser for visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize RL lane allocation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (without .zip extension)",
    )

    # Visualization
    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument("--fps", type=int, help="Target FPS")
    vis_group.add_argument("--width", type=int, help="Window width")
    vis_group.add_argument("--height", type=int, help="Window height")
    vis_group.add_argument(
        "--record",
        action="store_true",
        help="Record video",
    )

    # Environment
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--spawn-rate", type=float, help="Vehicle spawn rate")
    env_group.add_argument("--num-lanes", type=int, help="Number of lanes")
    env_group.add_argument("--episodes", type=int, help="Number of episodes to run")
    env_group.add_argument(
        "--lane-diversity",
        action="store_true",
        help="Cycle through multiple lane configurations [3,4,5,7]",
    )

    return parser


def load_config(config_path: str | Path | None = None) -> RLConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default.

    Returns:
        RLConfig instance
    """
    return RLConfig(config_path)
