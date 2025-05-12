from pathlib import Path

import torch

from shared_src.common import Config

from .core import logger


class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 0.0):
        """
        Early stopping class.

        Args:
            patience (int): Number of epochs to wait after the last improvement.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.val_loss_min = float("inf")

    @property
    def early_stop(self) -> bool:
        """
        Check if early stopping should occur.

        Returns:
            bool: True if early stopping should occur, False otherwise.
        """
        return self.counter >= self.patience

    @property
    def best_model_path(self) -> Path:
        """Get the path to the best model checkpoint."""
        file_path = Path(
            Config.get("global_cache_dir"),
            "lane_allocation",
            "checkpoints",
            "model_best.pth",
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> bool:
        """
        Call this function to check if early stopping should occur.

        Args:
            epoch (int): Current epoch number.
            val_loss (float): Current validation loss.
            model (nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            criterion (nn.Module): The loss function used.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(epoch, val_loss, model, optimizer, criterion)
        elif score <= self.best_score + self.delta:
            self.counter += 1
        else:
            self.best_score = score
            self._save_checkpoint(epoch, val_loss, model, optimizer, criterion)
            self.counter = 0

        return self.early_stop

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> None:
        """
        Saves the model when validation loss improves and meets save frequency.

        Args:
            epoch (int): Current epoch number.
            val_loss (float): Current validation loss.
            model (nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            criterion (nn.Module): The loss function used.
        """
        logger.debug(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
        )

        # Save model checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion": criterion.state_dict(),
            },
            self.best_model_path,
        )

        self.val_loss_min = val_loss
