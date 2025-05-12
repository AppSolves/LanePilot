import torch

# def build_edge_index(x, tolerance):
#     num_vehicles = x.size(0)
#     edge_index = []
#     for src in range(num_vehicles):
#         for dst in range(num_vehicles):
#             if src != dst:
#                 same_lane = abs(x[src][0] - x[dst][0]) < tolerance
#                 if same_lane:
#                     # Randomly select one of the two edges to avoid duplicates
#                     edge_index.append([src, dst])

#     if len(edge_index) == 0:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#     else:
#         edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()

#     return edge_index


def build_edge_index(
    x: torch.Tensor,
    lane_tolerance: int = 1,
    max_distance: float = 10.0,
    bidirectional: bool = True,
    return_weights: bool = False,
    weight_type: str = "inverse",  # "inverse", "linear", or "none"
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Generate edge indices for a graph based on vehicle positions and lane IDs.

    Args:
        x (torch.Tensor): Tensor of shape (N, D) where N is the number of vehicles
            and D is the number of features. The first column should be lane IDs,
            and the fourth column should be positions.
        lane_tolerance (int): Maximum difference in lane IDs to consider vehicles
            as connected.
        max_distance (float): Maximum distance between vehicles to consider them
            as connected.
        bidirectional (bool): If True, create edges in both directions.
        return_weights (bool): If True, return edge weights based on distance.
        weight_type (str): Type of weight calculation. Options are "inverse",
            "linear", or "none".
    Returns:
        tuple[torch.Tensor, torch.Tensor] | torch.Tensor: If return_weights is True,
            returns a tuple of edge indices and weights. Otherwise, returns only
            the edge indices.
    """

    lane_ids = x[:, 0].view(-1, 1)  # (N, 1)
    positions = x[:, 3].view(-1, 1)  # (N, 1)

    lane_diff = torch.abs(lane_ids - lane_ids.T)
    pos_diff = torch.abs(positions - positions.T)

    mask = (lane_diff <= lane_tolerance) & (pos_diff <= max_distance)
    mask.fill_diagonal_(False)

    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)  # (2, E)

    if bidirectional:
        reversed_edges = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([edge_index, reversed_edges], dim=1)

    if return_weights:
        distances = pos_diff[src, dst]
        if weight_type == "inverse":
            weights = 1.0 / (distances + 1e-6)  # Numeric stability
        elif weight_type == "linear":
            weights = max_distance - distances
        else:
            weights = torch.ones_like(distances)

        if bidirectional:
            weights = torch.cat([weights, weights], dim=0)

        return edge_index, weights

    return edge_index
