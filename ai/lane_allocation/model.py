from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

from .core import logger
from .vehicle_state import NUM_LANES


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


class LaneAllocationGAT(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = NUM_LANES,
        heads: int = 2,
    ):
        super().__init__()
        self.input_layer = GATv2Conv(input_dim, hidden_dim, heads=heads)
        self.hidden_layer_1 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads)
        self.hidden_layer_2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads)
        self.output_layer = Linear(hidden_dim * heads, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.input_layer(x, edge_index)
        x = F.relu(x)
        x = self.hidden_layer_1(x, edge_index)
        x = F.relu(x)
        x = self.hidden_layer_2(x, edge_index)
        x = F.relu(x)
        x = self.output_layer(x)
        return x  # Pooling is not needed since we are classifying each vehicle

    def test(self, test_loader: DataLoader, device: torch.device):
        """Evaluate the model on the test data."""
        logger.debug("Evaluating the model...")
        self.to(device)
        self.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = self(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct_preds += (pred == batch.y).sum().item()
                total_preds += batch.y.size(0)

        return (correct_preds / total_preds) * 100

    def inference(
        self,
        model_path: str | Path,
        device: torch.device,
    ):
        """Load the model for inference."""
        logger.debug(f"Loading model from '{model_path}'")
        state_dict = torch.load(model_path, weights_only=True, map_location=device).get(
            "model_state_dict"
        )
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        logger.debug("Model loaded for inference.")
        return self
