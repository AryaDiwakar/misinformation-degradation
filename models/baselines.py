import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GCNModel(nn.Module):
    """Standard GCN baseline using PyG's GCNConv."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        graph_h = x.mean(dim=0, keepdim=True)
        return self.classifier(graph_h)


class GATModel(nn.Module):
    """Graph Attention Network using PyG's GATConv."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GATConv(input_dim, hidden_dim // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.GATConv(hidden_dim, hidden_dim // heads, heads=heads))
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        graph_h = x.mean(dim=0, keepdim=True)
        return self.classifier(graph_h)


class BiGCNModel(nn.Module):
    """BiGCN-style using PyG's GCNConv with top-down and bottom-up branches."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.td_convs = nn.ModuleList()
        self.bu_convs = nn.ModuleList()
        self.td_convs.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        self.bu_convs.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.td_convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
            self.bu_convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        td_x = x
        for conv in self.td_convs:
            td_x = conv(td_x, edge_index)
            td_x = F.relu(td_x)
            td_x = F.dropout(td_x, self.dropout, training=self.training)

        # Bottom-up: reverse edge direction
        rev_edge = edge_index.flip(0)
        bu_x = x
        for conv in self.bu_convs:
            bu_x = conv(bu_x, rev_edge)
            bu_x = F.relu(bu_x)
            bu_x = F.dropout(bu_x, self.dropout, training=self.training)

        td_pool = td_x.mean(dim=0, keepdim=True)
        bu_pool = bu_x.mean(dim=0, keepdim=True)
        h = torch.cat([td_pool, bu_pool], dim=-1)
        return self.classifier(h)
