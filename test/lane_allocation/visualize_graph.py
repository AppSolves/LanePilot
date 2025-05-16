import argparse
import random as rd
from math import cbrt
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from torch_geometric.data import Data

from ai.lane_allocation import MODULE_CONFIG as GAT_CONFIG
from ai.lane_allocation.model import DatasetSplit
from ai.lane_allocation.train import load_dataset_split
from shared_src.data_preprocessing import unpack_dataset


def plot_graph(index: int, data: Data, seed: int, num_lanes: int):
    # Convert edge_index to numpy for easier handling
    edge_index = data.edge_index.cpu().numpy()
    G = nx.DiGraph()
    # Add nodes with optional feature labels
    for i in range(data.x.size(0)):
        G.add_node(i, label=f"V{i}")
    # Add edges
    dst_counter = {}
    for src, dst in edge_index.T:
        dst_counter[dst] = dst_counter.get(dst, 0) + 1
        G.add_edge(src, dst)

    # Add three big nodes on the right (lane nodes)
    lane_nodes = []
    lane_offset = 0.07  # Offset for right placement
    for lane in range(num_lanes):
        lane_node = f"Lane{lane}"
        lane_nodes.append(lane_node)
        G.add_node(lane_node, label=f"Lane {lane}")

    # Connect each vehicle node to its lane node according to data.y
    for i in range(data.x.size(0)):
        lane = int(data.y[i].item())
        G.add_edge(i, f"Lane{lane}")

    # Draw the graph
    plt.figure(figsize=(8, 4.5))  # Stretch y-axis
    pos = nx.spring_layout(G, seed=seed, k=1 / cbrt(len(G.nodes())))  # Initial layout

    # Move lane nodes to the right
    for lane in range(num_lanes):
        max_x = max(pos[i][0] for i in range(data.x.size(0)))
        pos[f"Lane{lane}"] = (
            max_x + lane_offset,
            lane_offset - (lane + 1) * lane_offset,  # Vertical placement
        )

    labels = nx.get_node_attributes(G, "label")
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if str(n).startswith("Lane"):
            node_colors.append("orange")
            node_sizes.append(2000)
        else:
            node_colors.append("skyblue")
            try:
                maximum = max(dst_counter)
            except ValueError:
                maximum = 1
            node_sizes.append(750 * (dst_counter.get(n, 1) / maximum))

    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        edge_color="gray",
        font_family=plt.rcParams["font.family"],
        node_size=node_sizes,
    )
    ax = plt.gca()
    if ax.collections:
        ax.collections[0].set_edgecolor("#000000")
    title = f"Graph Visualization | Graph {index} | Seed {seed}"
    plt.title(title)
    legend_elements = [
        Patch(facecolor="skyblue", edgecolor="k", label="Vehicle"),
        Patch(facecolor="orange", edgecolor="k", label="Lane"),
        Patch(facecolor="gray", edgecolor="gray", label="Edge (Relationship)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    if manager := plt.gcf().canvas.manager:
        manager.set_window_title(title)  # Set window title

    # --- Select one random vehicle and display its features ---
    num_vehicles = data.x.size(0)
    random_idx = np.random.randint(0, num_vehicles)
    features = data.x[random_idx].cpu().numpy()
    feature_str = "\n".join([f"f{i}: {v:.3f}" for i, v in enumerate(features)])
    textstr = f"Vehicle {random_idx} features:\n{feature_str}"

    # Place the textbox at the right bottom
    plt.gcf().text(
        0.98,
        0.85,
        textstr,
        fontsize=12,
        va="top",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        fontfamily=plt.rcParams["font.family"],
    )

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Graphs")
    parser.add_argument(
        "-ng",
        "--num_graphs",
        type=int,
        default=1,
        help="Number of graphs to visualize from the test set",
    )
    parser.add_argument(
        "-omd",
        "--override_max_distance",
        type=float,
        default=None,
        help="Override the max distance for vehicle settings",
    )
    parser.add_argument(
        "-ff",
        "--font_family",
        type=str,
        default="sans-serif",
        help="Font family for matplotlib text",
    )
    args = parser.parse_args()
    num_graphs = args.num_graphs
    override_max_distance = args.override_max_distance
    plt.rcParams["font.family"] = args.font_family

    # Set seed for reproducibility
    environment = GAT_CONFIG.get("environment")
    num_lanes = environment.get("num_lanes")
    seed = environment.get("seed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rd.seed(seed)

    dataset_path = Path(GAT_CONFIG.get("dataset_path"))
    dataset_path = unpack_dataset(dataset_path, "lane_allocation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vehicle_settings = GAT_CONFIG.get("vehicle", {})
    max_distance_cm = override_max_distance or vehicle_settings.get("max_distance_cm")

    # Load the dataset
    test_set = load_dataset_split(
        dataset_path, DatasetSplit.TEST, device, max_distance_cm
    )

    # Evaluate the model
    for index, graph in enumerate(test_set[:num_graphs]):
        plot_graph(index, graph, seed, num_lanes)


if __name__ == "__main__":
    main()
