from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

from ai.lane_allocation import LaneAllocationGAT, logger
from ai.lane_allocation.train import DatasetSplit, load_dataset_split
from shared_src.common import Config
from shared_src.data_preprocessing import unpack_dataset
from shared_src.inference import MODULE_CONFIG as GAT_CONFIG
from shared_src.inference import NUM_LANES


def plot_confusion_matrix(y_true, y_pred, classes, title=None):
    """
    Plot the confusion matrix using seaborn heatmap.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        classes (list): List of class names.
        title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title if title else "Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    logger.debug("Plotting confusion matrix...")


def evaluate_model(
    model: LaneAllocationGAT,
    test_loader: DataLoader,
    device: torch.device,
):
    """
    Evaluate the model on the test data and plot the confusion matrix.

    Args:
        model: The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on.
    """
    logger.debug("Evaluating the model...")
    model.device = device
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Plot confusion matrix
    classes = [f"Lane {i}" for i in range(NUM_LANES)]
    plot_confusion_matrix(
        all_labels, all_preds, classes, title="Confusion Matrix for Lane Allocation"
    )
    logger.debug("Model evaluation completed.")


def main():
    # Load the model
    model_config = GAT_CONFIG.get("model", {})

    model_path = Path(
        Config.get("global_assets_dir"),
        "trained_models",
        "lane_allocation",
        "lane_allocation.pt",
    )
    dataset_config = GAT_CONFIG.get("dataset", {})
    dataset_path = Path(dataset_config.get("path"))
    dataset_path = unpack_dataset(dataset_path, "lane_allocation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = model_config.get("batch_size")

    vehicle_settings = GAT_CONFIG.get("vehicle", {})
    max_distance_cm = vehicle_settings.get("max_distance_cm")

    # Assuming test_loader is already defined and loaded with test data
    test_loader = DataLoader(
        load_dataset_split(dataset_path, DatasetSplit.TEST, max_distance_cm),
        batch_size=batch_size,
    )

    # Create the model instance
    gat_config = GAT_CONFIG.get("model", {})
    num_heads = gat_config.get("num_heads")
    hidden_dim = gat_config.get("hidden_dim")
    input_dim = 4 + NUM_LANES

    model = LaneAllocationGAT(
        input_dim=input_dim, hidden_dim=hidden_dim, heads=num_heads
    )

    # Load the model state
    model.inference(model_path, device)

    # Evaluate the model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
