import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm, global_max_pool


class GraphAwareNorm(nn.Module):
    def __init__(self, hidden_dim, normalization="batchnorm", graphnorm_fallback=None):
        super().__init__()
        norm_type = (normalization or "batchnorm").lower()
        aliases = {
            "batch": "batchnorm",
            "bn": "batchnorm",
            "graph": "graphnorm",
            "gn": "graphnorm",
            "layer": "layernorm",
            "ln": "layernorm",
            "identity": "none",
        }
        norm_type = aliases.get(norm_type, norm_type)
        if norm_type == "graphnorm" and graphnorm_fallback is not None:
            norm_type = graphnorm_fallback
        self.norm_type = norm_type
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = GraphNorm(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(
                "normalization must be one of: batchnorm, graphnorm, layernorm, none"
            )

    def forward(self, x, batch=None):
        if self.norm_type == "graphnorm":
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            return self.norm(x, batch)
        return self.norm(x)


class NormedTwoLayerMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout_rate,
        normalization="batchnorm",
        graphnorm_fallback=None,
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = GraphAwareNorm(hidden_dim, normalization, graphnorm_fallback)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = GraphAwareNorm(hidden_dim, normalization, graphnorm_fallback)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, batch=None):
        x = self.linear1(x)
        x = self.norm1(x, batch)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x, batch)
        return self.activation(x)

class EdgeBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        edge_dim,
        hidden_dim,
        dropout_rate,
        normalization="graphnorm",
    ):
        super().__init__()
        self.edge_mlp = NormedTwoLayerMLP(
            input_dim * 2 + edge_dim,
            hidden_dim,
            dropout_rate,
            normalization,
        )

    def forward(self, src, dest, edge_attr, edge_batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out, edge_batch)

class GlobalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        global_dim,
        dropout_rate,
        normalization="batchnorm",
    ):
        super().__init__()
        self.global_mlp = NormedTwoLayerMLP(
            hidden_dim + global_dim,
            hidden_dim,
            dropout_rate,
            normalization,
            graphnorm_fallback="layernorm",
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, u, batch):
        u_x = global_max_pool(x, batch)
        out = self.global_mlp(torch.cat([u_x, u], dim=1))
        return self.output(out)
