import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionPolicyNetwork(BaseFeaturesExtractor):
    """
    Custom feature extractor using self-attention for vehicle interactions.
    This replaces the GATv2 approach with an RL-compatible attention mechanism.
    Can be used with PPO, A2C, or other SB3 algorithms.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        num_attention_heads: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        # Feature embedding layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention for capturing vehicle interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        # Feed-forward layers
        self.ff_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.ff_layer2 = nn.Linear(hidden_dim, features_dim)

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention-based feature extractor.

        Args:
            observations: Tensor of shape (batch_size, obs_dim)

        Returns:
            features: Tensor of shape (batch_size, features_dim)
        """
        # Embed input
        x = F.relu(self.input_layer(observations))

        # Add batch dimension if single observation
        if x.dim() == 2:
            x_att = x.unsqueeze(1)  # (batch, 1, hidden)
        else:
            x_att = x

        # Self-attention (vehicle attends to its own state features)
        attn_output, _ = self.attention(x_att, x_att, x_att)

        # Residual connection + normalization
        x = self.norm1(x + attn_output.squeeze(1))

        # Feed-forward
        ff_out = F.relu(self.ff_layer1(x))
        ff_out = self.ff_layer2(ff_out)

        # Final residual + normalization
        features = self.norm2(x + ff_out)

        return features


class SimpleMLPExtractor(BaseFeaturesExtractor):
    """
    Simple MLP baseline for comparison.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
