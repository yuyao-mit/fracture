
# https://arxiv.org/abs/2003.03485

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x

class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ker_width):
        super(GraphConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_net = DenseNet([3, ker_width//2, ker_width, in_channels * out_channels], torch.nn.ReLU)
        self.root_weight = nn.Parameter(torch.randn(in_channels, out_channels))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.gaussian_param = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        x_source = x[source_nodes]
        distances = edge_attr[:, 0:1]
        rel_pos = edge_attr[:, 1:3]
        gaussian_weights = torch.exp(-distances**2 / (self.gaussian_param**2 + 1e-8))
        gaussian_weights = gaussian_weights.repeat(1, self.out_channels)
        messages = torch.matmul(x_source, self.root_weight)
        messages = messages * gaussian_weights
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)
        out.index_add_(0, target_nodes, messages)
        out = out + torch.matmul(x, self.root_weight)
        out = out + self.bias
        return out

class PhaseFieldPredictor(torch.nn.Module):
    def __init__(self, width=64, ker_width=32, depth=4, ker_in=3, in_channels=10, out_channels=10, 
                 input_time_steps=5, grid_size=64):
        super(PhaseFieldPredictor, self).__init__()
        self.depth = depth
        self.input_time_steps = input_time_steps
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_encoder = nn.LSTM(in_channels, width//2, num_layers=2, batch_first=True)
        self.fc1 = torch.nn.Linear(width//2, width)
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(width, width, ker_width) for _ in range(depth)
        ])
        self.fc2 = torch.nn.Linear(width, ker_width)
        self.fc3 = torch.nn.Linear(ker_width, out_channels)
        self.register_buffer('edge_index', self._build_grid_edges(grid_size))
        self.register_buffer('edge_attr', self._build_edge_attr(grid_size))
    
    def _build_grid_edges(self, grid_size):
        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                center_idx = i * grid_size + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            neighbor_idx = ni * grid_size + nj
                            edges.append([center_idx, neighbor_idx])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _build_edge_attr(self, grid_size):
        edge_attrs = []
        for edge in self.edge_index.t():
            src, dst = edge[0].item(), edge[1].item()
            src_i, src_j = src // grid_size, src % grid_size
            dst_i, dst_j = dst // grid_size, dst % grid_size
            dist = np.sqrt((src_i - dst_i)**2 + (src_j - dst_j)**2)
            rel_i, rel_j = dst_i - src_i, dst_j - src_j
            edge_attrs.append([dist, rel_i, rel_j])
        return torch.tensor(edge_attrs, dtype=torch.float32)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(B, H*W, T, C)
        x_encoded = []
        for b in range(B):
            encoded, _ = self.temporal_encoder(x[b])
            encoded = encoded[:, -1, :]
            x_encoded.append(encoded)
        x = torch.stack(x_encoded, dim=0)
        x = self.fc1(x)
        x = F.relu(x)
        outputs = []
        for b in range(B):
            node_features = x[b]
            for k in range(self.depth):
                node_features = self.conv_layers[k](node_features, self.edge_index, self.edge_attr)
                if k != self.depth - 1:
                    node_features = F.relu(node_features)
            outputs.append(node_features)
        x = torch.stack(outputs, dim=0)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = x.reshape(B, H, W, self.out_channels)
        x = x.permute(0, 3, 1, 2)
        x = x.unsqueeze(1)
        return x
