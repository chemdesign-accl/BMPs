import torch.nn as nn
import torch
from .blocks import EdgeBlock, GlobalBlock
from .node_blocks import (
    BMPNodeBlock, ABMPNodeBlock, CBMPNodeBlock,
    BMP_SNNodeBlock, ABMP_SNNodeBlock, UMPNodeBlock
)
class InteractionNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        edge_dim,
        hidden_dim,
        global_dim,
        dropout_rate,
        variant='BMP',
        normalization="batchnorm",
        message_passing_steps=1,
    ):
        super().__init__()
        self.variant = variant
        self.normalization = normalization
        self.message_passing_steps = int(message_passing_steps)
        if self.message_passing_steps < 1:
            raise ValueError("message_passing_steps must be >= 1.")
        self.edge_models = nn.ModuleList()
        self.node_models = nn.ModuleList()
        for step in range(self.message_passing_steps):
            step_input_dim = input_dim if step == 0 else hidden_dim
            self.edge_models.append(
                EdgeBlock(
                    step_input_dim,
                    edge_dim,
                    hidden_dim,
                    dropout_rate,
                    normalization,
                )
            )
            self.node_models.append(
                self._make_node_model(
                    step_input_dim,
                    edge_dim,
                    hidden_dim,
                    dropout_rate,
                    normalization,
                )
            )
        self.global_model = GlobalBlock(
            hidden_dim, global_dim, dropout_rate, normalization
        )
    def _make_node_model(
        self,
        input_dim,
        edge_dim,
        hidden_dim,
        dropout_rate,
        normalization,
    ):
        if self.variant == 'BMP':
            return BMPNodeBlock(hidden_dim, dropout_rate, normalization)
        elif self.variant == 'ABMP':
            return ABMPNodeBlock(
                input_dim, edge_dim, hidden_dim, dropout_rate, normalization
            )
        elif self.variant == 'CBMP':
            return CBMPNodeBlock(hidden_dim, dropout_rate, normalization)
        elif self.variant == 'BMP+SN':
            return BMP_SNNodeBlock(
                input_dim, hidden_dim, dropout_rate, normalization
            )
        elif self.variant == 'ABMP+SN':
            return ABMP_SNNodeBlock(
                input_dim, edge_dim, hidden_dim, dropout_rate, normalization
            )
        elif self.variant == 'UMP':
            return UMPNodeBlock(
                input_dim, hidden_dim, dropout_rate, normalization
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def forward(self, x, edge_index, edge_attr, u, batch, norm=None):
        node_score_steps = []
        for edge_model, node_model in zip(self.edge_models, self.node_models):
            src = x[edge_index[0]]
            dest = x[edge_index[1]]
            edge_batch = batch[edge_index[0]] if batch is not None and edge_index.numel() else None
            message = edge_model(src, dest, edge_attr, edge_batch=edge_batch)
            if self.variant == 'CBMP':
                from torch_geometric.utils import degree
                def compute_norm(edge_index, num_nodes):
                    row, col = edge_index
                    deg = degree(row, num_nodes=num_nodes)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                    return norm
                norm = compute_norm(edge_index, x.size(0))
                x, x_weights = node_model(x, edge_index, message, norm, batch=batch)
            elif self.variant in ['ABMP', 'ABMP+SN']:
                x, x_weights = node_model(
                    x, edge_index, edge_attr, message, batch=batch
                )
            else:
                x, x_weights = node_model(x, edge_index, message, batch=batch)
            node_score_steps.append(x_weights)

        u = self.global_model(x, u, batch)
        if len(node_score_steps) == 1:
            return u, node_score_steps[0]
        x_weights = torch.stack(node_score_steps, dim=0).mean(dim=0)
        return u, x_weights
