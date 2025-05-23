import shutil
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from shared_src.data_preprocessing import (
    DatasetSplit,
    load_dataset_split,
    unpack_dataset,
)
from shared_src.inference import MAX_VEHICLES_PER_LANE, NUM_LANES
from shared_src.postprocessing import export_model_to_trt

from .core import MODULE_CONFIG, Config, logger
from .early_stopping import EarlyStopping
from .model import LaneAllocationGAT


def train():
    """Main function to train the GAT."""
    # Set seed for reproducibility
    environment = MODULE_CONFIG.get("environment")
    seed = environment.get("seed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_config = MODULE_CONFIG.get("dataset", {})
    dataset_path = Path(dataset_config.get("path"))
    logger.debug(f"Dataset path: {dataset_path}")

    if not dataset_path:
        logger.error("Dataset path is not provided.")
        raise ValueError("Dataset path must be provided.")

    model_config = MODULE_CONFIG.get("model", {})
    num_epochs = model_config.get("num_epochs")
    batch_size = model_config.get("batch_size")
    num_heads = model_config.get("num_heads")
    hidden_dim = model_config.get("hidden_dim")

    vehicle_settings = MODULE_CONFIG.get("vehicle", {})
    max_distance_cm = vehicle_settings.get("max_distance_cm")

    dataset_path = unpack_dataset(dataset_path, "lane_allocation")
    train_dataset = load_dataset_split(
        dataset_path, DatasetSplit.TRAIN, device, max_distance_cm
    )
    val_dataset = load_dataset_split(
        dataset_path, DatasetSplit.VALIDATION, device, max_distance_cm
    )
    test_dataset = load_dataset_split(
        dataset_path, DatasetSplit.TEST, device, max_distance_cm
    )

    # Get class weights for the dataset
    class_counts = [0] * NUM_LANES
    for data in train_dataset:
        for label in data.y:
            class_counts[label.item()] += 1
    total_preds = sum(class_counts)
    class_weights = {
        i: 100 * round(count / total_preds, 3) for i, count in enumerate(class_counts)
    }
    label_weights = torch.tensor(
        [100 / value for value in class_weights.values()], dtype=torch.float32
    )
    label_weights = label_weights / label_weights.mean()
    logger.debug(f"Class counts (in %): {class_weights} | Weights: {label_weights}")

    num_features = train_dataset[0].x.shape[1]

    early_stopping_config = MODULE_CONFIG.get("early_stopping", {})
    patience = early_stopping_config.get("patience")

    optimizer_config = MODULE_CONFIG.get("optimizer", {})
    learning_rate = optimizer_config.get("learning_rate")
    weight_decay = optimizer_config.get("weight_decay")
    epsilon = optimizer_config.get("epsilon")
    t_0 = optimizer_config.get("t_0")
    t_mult = optimizer_config.get("t_mult")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = LaneAllocationGAT(
        input_dim=num_features, hidden_dim=hidden_dim, heads=num_heads
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=epsilon
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t_0, T_mult=t_mult, eta_min=learning_rate / 100
    )
    criterion = torch.nn.CrossEntropyLoss(weight=label_weights.to(device)).to(device)
    early_stopping = EarlyStopping(patience)

    logger.info("🔧 Training starting...\n")
    iters = len(train_loader)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs: torch.Tensor = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(outputs, batch.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch + i / iters)

        # Validation phase
        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(outputs, batch.y)
                val_loss += loss.item()

                prediction = outputs.argmax(dim=1)
                correct_preds += (prediction == batch.y).sum().item()
                total_preds += batch.y.size(0)

        accuracy = (correct_preds / total_preds) * 100

        logger.debug(
            f"Epoch {epoch:2d} | Training Loss: {total_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {accuracy:.2f}%"
        )

        # Check early stopping
        if early_stopping(epoch, val_loss, model, optimizer, criterion):
            logger.info("⏹️  Early stopping triggered.")
            break

    # Save the model
    logger.info("Saving the model...")
    best_model_path = early_stopping.best_model_path
    model_save_path = Path(
        Config.get("global_assets_dir"),
        "trained_models",
        "lane_allocation",
        "lane_allocation.pt",
    )
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        best_model_path,
        model_save_path,
    )
    logger.info(f"Model saved to {model_save_path}")

    # Load the best model checkpoint again and test it
    model.inference(
        model_save_path,
        device,
    )

    accuracy = model.test(test_loader)
    logger.info(f"Accuracy on test split: {accuracy:.2f}%\n")
    logger.info("✅ Training completed successfully! 🚀")

    # Export the model to ONNX format
    dummy_input = (
        torch.randn(test_dataset[0].x.shape).to(device),
        test_dataset[0].edge_index.to(device).long(),
    )
    export_model_to_trt(
        model,
        Path(
            Config.get("global_assets_dir"),
            "trained_models",
            "lane_allocation",
            "lane_allocation.onnx",
        ),
        dummy_input,
        input_names=["x", "edge_index"],
        output_names=["output"],
        dynamic_axes={
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "output": {0: "num_nodes"},
        },
        shapes={
            "min_shapes": f"x:1x{num_features},edge_index:2x1",
            "opt_shapes": f"x:{MAX_VEHICLES_PER_LANE}x{num_features},edge_index:2x{MAX_VEHICLES_PER_LANE * 2}",
            "max_shapes": f"x:{MAX_VEHICLES_PER_LANE ** 2}x{num_features},edge_index:2x{(MAX_VEHICLES_PER_LANE * 2) ** 2}",
        },
    )


if __name__ == "__main__":
    train()
