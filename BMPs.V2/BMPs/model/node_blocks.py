import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import softmax
from .blocks import NormedTwoLayerMLP

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.W_src = nn.Linear(input_dim, hidden_dim, bias=False)  
        self.W_dest = nn.Linear(input_dim, hidden_dim, bias=False) 
        self.W_edge = nn.Linear(edge_dim, hidden_dim, bias=False) 
        self.attn_vector = nn.Parameter(torch.Tensor(1, hidden_dim)) 
        nn.init.xavier_uniform_(self.attn_vector)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, src, dest, edge_attr, edge_index):   
        src_transformed = self.W_src(src) 
        dest_transformed = self.W_dest(dest) 
        edge_transformed = self.W_edge(edge_attr)  
        edge_scores = self.leaky_relu(
            torch.matmul(src_transformed + dest_transformed + edge_transformed, self.attn_vector.t())
        ) 
        _, col = edge_index 
        attn_weights = softmax(edge_scores, col)
        return attn_weights

def message_node_scores(edge_index, message, num_nodes, edge_weights=None):
    row, col = edge_index
    edge_scores = message.norm(p=2, dim=1)
    if edge_weights is not None:
        edge_scores = edge_scores * edge_weights.view(-1)
    incoming = scatter(edge_scores, col, dim=0, dim_size=num_nodes, reduce='max')
    outgoing = scatter(edge_scores, row, dim=0, dim_size=num_nodes, reduce='max')
    return 0.5 * (incoming + outgoing)

class BMPNodeBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate, normalization="batchnorm"):
        super().__init__()
        self.node_mlp = NormedTwoLayerMLP(
            2 * hidden_dim, hidden_dim, dropout_rate, normalization
        )
    def forward(self, x, edge_index, message, batch=None):
        row, col = edge_index
        node_scores = message_node_scores(edge_index, message, x.size(0))
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([forward, backward], dim=1)
        out = self.node_mlp(out, batch)
        return out, node_scores
class UMPNodeBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, normalization="batchnorm"):
        super().__init__()
        self.node_mlp_1 = NormedTwoLayerMLP(
            input_dim + hidden_dim, hidden_dim, dropout_rate, normalization
        )
        self.node_mlp_2 = NormedTwoLayerMLP(
            input_dim + hidden_dim, hidden_dim, dropout_rate, normalization
        )
    def forward(self, x, edge_index, message, batch=None):
        row, col = edge_index
        node_scores = message_node_scores(edge_index, message, x.size(0))
        out = x[row]  
        out = torch.cat([message, out], dim=1)
        edge_batch = batch[row] if batch is not None else None
        out = self.node_mlp_1(out, edge_batch)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out, batch)
        return out, node_scores
class ABMPNodeBlock(nn.Module): 
    def __init__(self, input_dim, edge_dim, hidden_dim, dropout_rate, normalization="graphnorm"):
        super().__init__()
        self.attention = AttentionMechanism(input_dim, edge_dim, hidden_dim)
        self.node_mlp = NormedTwoLayerMLP(
            2 * hidden_dim, hidden_dim, dropout_rate, normalization
        )
    def forward(self, x, edge_index, edge_attr, message, batch=None):
        row, col = edge_index
        src, dest = x[row], x[col]
        attn_weights = self.attention(src, dest, edge_attr, edge_index)
        node_scores = message_node_scores(
            edge_index, message, x.size(0), edge_weights=attn_weights
        )
        message = message * attn_weights.view(-1, 1)
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([forward, backward], dim=1)
        out = self.node_mlp(out, batch)
        return out, node_scores


class CBMPNodeBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate, normalization="batchnorm"):
        super().__init__()
        self.node_mlp = NormedTwoLayerMLP(
            2 * hidden_dim, hidden_dim, dropout_rate, normalization
        )
    def forward(self, x, edge_index, edge_attr, norm, batch=None):
        row, col = edge_index
        edge_attr = norm.unsqueeze(1) * edge_attr
        node_scores = message_node_scores(edge_index, edge_attr, x.size(0))
        forward = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(edge_attr, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([forward, backward], dim=1)
        out = self.node_mlp(out, batch)
        return out, node_scores

class BMP_SNNodeBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, normalization="batchnorm"):
        super().__init__()
        self.node_mlp = NormedTwoLayerMLP(
            input_dim + 2 * hidden_dim,
            hidden_dim,
            dropout_rate,
            normalization,
        )
    def forward(self, x, edge_index, message, batch=None):
        row, col = edge_index
        node_scores = message_node_scores(edge_index, message, x.size(0))
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([x, forward, backward], dim=1)
        out = self.node_mlp(out, batch)
        return out, node_scores
class ABMP_SNNodeBlock(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, dropout_rate, normalization="batchnorm"):
        super().__init__()
        self.attention = AttentionMechanism(input_dim, edge_dim, hidden_dim)
        self.node_mlp = NormedTwoLayerMLP(
            2 * hidden_dim + input_dim,
            hidden_dim,
            dropout_rate,
            normalization,
        )
    def forward(self, x, edge_index, edge_attr, message, batch=None):
        row, col = edge_index
        src, dest = x[row], x[col]
        attn_weights = self.attention(src, dest, edge_attr, edge_index)
        node_scores = message_node_scores(
            edge_index, message, x.size(0), edge_weights=attn_weights
        )
        message = message * attn_weights.view(-1, 1)
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([x, forward, backward], dim=1)
        out = self.node_mlp(out, batch)
        return out, node_scores
