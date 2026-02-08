"""
Export trained PPO lane allocation model to ONNX and TensorRT formats.
Supports multi-vehicle centralized control (v2 architecture).
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

# Suppress benign ONNX export warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.onnx")
warnings.filterwarnings("ignore", message=".*Missing annotation.*")

from shared_src.postprocessing import export_model_to_trt

from .config_utils import load_config
from .core import logger


class PolicyWrapper(torch.nn.Module):
    """
    Wrapper to extract and export just the policy network from PPO.
    Converts Stable-Baselines3 PPO policy to a standalone PyTorch model.
    """

    def __init__(self, ppo_model: PPO):
        super().__init__()
        self.policy = ppo_model.policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.

        Args:
            observation: Tensor of shape [batch_size, obs_dim] or [obs_dim]
                        For multi-vehicle v2: obs_dim = 81 (15 vehicles * 5 features + 6 global)

        Returns:
            action_logits: Tensor of shape [batch_size, action_dim]
                          For multi-vehicle v2: action_dim = 45 (15 vehicles * 3 actions)
        """
        # Ensure batch dimension
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        # Get action distribution from policy
        with torch.no_grad():
            # PPO policy returns distribution, we want logits
            features = self.policy.extract_features(observation)
            latent_pi, _ = self.policy.mlp_extractor(features)
            action_logits = self.policy.action_net(latent_pi)

        return action_logits


def export_lane_allocation_model(
    model_path: str,
    output_dir: str,
    config: Any,
    export_trt: bool = True,
) -> tuple[Path, Path | None]:
    """
    Export PPO lane allocation model to ONNX and optionally TensorRT.

    Args:
        model_path: Path to trained PPO model (.zip file)
        output_dir: Directory to save exported models
        config: RL configuration dict
        export_trt: Whether to also export TensorRT engine

    Returns:
        Tuple of (onnx_path, trt_path)
    """
    logger.info(f"Loading PPO model from {model_path}")

    # Load trained model
    if not Path(model_path + ".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    # Load model on CPU for export (more compatible)
    model = PPO.load(model_path, device="cpu")
    logger.info("✓ Model loaded successfully")

    # Get environment config for observation space
    env_config = config.get("environment", {})
    multi_vehicle = env_config.get("multi_vehicle_control", True)
    max_vehicles = env_config.get("max_vehicles", 15)

    # Calculate observation and action dimensions
    if multi_vehicle:
        obs_dim = max_vehicles * 5 + 6  # Vehicle features + global features
        action_dim = max_vehicles * 3  # 3 actions per vehicle (keep/left/right)
    else:
        obs_dim = 15  # Single vehicle observation
        action_dim = 3  # Single vehicle action

    logger.info(f"Observation dimension: {obs_dim}")
    logger.info(f"Action dimension: {action_dim}")

    # Wrap policy for export
    policy_wrapper = PolicyWrapper(model)
    policy_wrapper.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, obs_dim)

    # Output paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "lane_allocation"

    logger.info("Exporting to ONNX and TensorRT...")

    # Export to ONNX then TensorRT
    trt_path = None
    if export_trt:
        trt_path = export_model_to_trt(
            model=policy_wrapper,
            save_path=save_path,
            dummy_input=(dummy_input,),
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
            shapes={
                "min_shapes": f"observation:1x{obs_dim}",
                "opt_shapes": f"observation:1x{obs_dim}",
                "max_shapes": f"observation:16x{obs_dim}",  # Support batch inference
            },
        )
        logger.info(f"✓ TensorRT model exported to {trt_path}")
        onnx_path = save_path.with_suffix(".onnx")
    else:
        # ONNX only
        from shared_src.postprocessing import export_model_to_onnx

        onnx_path = export_model_to_onnx(
            model=policy_wrapper,
            save_path=save_path.with_suffix(".onnx"),
            dummy_input=(dummy_input,),
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
        )
        logger.info(f"✓ ONNX model exported to {onnx_path}")

    # Test exported model
    logger.info("Testing exported model...")
    test_obs = np.random.randn(obs_dim).astype(np.float32)

    # Get original model prediction
    original_action, _ = model.predict(test_obs, deterministic=True)
    logger.info(f"Original model action: {original_action}")

    # Test ONNX model
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("✓ ONNX model validation passed")

    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_output = ort_session.run(
        None,
        {"observation": test_obs.reshape(1, -1)},
    )[0]

    # Convert logits to actions
    onnx_logits = np.array(onnx_output)
    if multi_vehicle:
        # Reshape to [max_vehicles, 3] and get argmax per vehicle
        onnx_actions = onnx_logits.reshape(max_vehicles, 3).argmax(axis=1)
    else:
        onnx_actions = onnx_logits.argmax(axis=1)[0]

    logger.info(f"ONNX model action: {onnx_actions}")

    # Verify actions match (approximately, due to potential numerical differences)
    if multi_vehicle:
        match_rate = (original_action == onnx_actions).mean()
        logger.info(f"Action match rate: {match_rate * 100:.1f}%")
        if match_rate < 0.8:
            logger.warning("⚠️  Low action match rate - model may have export issues")
    else:
        if original_action == onnx_actions:
            logger.info("✓ Actions match!")
        else:
            logger.warning("⚠️  Actions don't match - check export")

    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info(f"ONNX model: {onnx_path}")
    if trt_path:
        logger.info(f"TensorRT engine: {trt_path}")
    logger.info("=" * 60)

    return onnx_path, trt_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export trained PPO lane allocation model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (without .zip extension)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./assets/trained_models/lane_allocation",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--no-trt",
        action="store_true",
        help="Skip TensorRT export (ONNX only)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Auto-detect model if not specified
    model_path = args.model
    if model_path is None:
        log_dir = Path(config.get("training.log_dir", "./runtime/logs"))

        # Look for most recent best_model
        import glob

        candidates = glob.glob(
            str(log_dir / "ppo_lane_allocation_*" / "best_model.zip")
        )

        if candidates:
            model_path = str(
                Path(max(candidates, key=os.path.getmtime)).with_suffix("")
            )
            logger.info(f"Auto-detected model: {model_path}")
        else:
            # Try final model
            candidates = glob.glob(
                str(log_dir / "ppo_lane_allocation_*" / "ppo_lane_final.zip")
            )
            if candidates:
                model_path = str(
                    Path(max(candidates, key=os.path.getmtime)).with_suffix("")
                )
                logger.info(f"Auto-detected model: {model_path}")
            else:
                raise FileNotFoundError(
                    "No trained model found. Train a model first:\n"
                    "  python -m ai.lane_allocation.train"
                )

    # Export
    export_lane_allocation_model(
        model_path=model_path,
        output_dir=args.output,
        config=config,
        export_trt=not args.no_trt,
    )


if __name__ == "__main__":
    main()
