import torch
import torch.nn.functional as F


class WeightedGraphSAGE(torch.nn.Module):
    def __init__(self, dim: int, dropout: float = 0.01):
        super().__init__()
        self.fc_self = torch.nn.Linear(dim, dim, bias=False)
        self.fc_neigh = torch.nn.Linear(dim, dim, bias=False)
        self.norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, weighted_adj: torch.Tensor) -> torch.Tensor:
        row_sum = weighted_adj.sum(dim=1, keepdim=True).clamp_min(1e-12)
        neigh = (weighted_adj @ x) / row_sum
        out = self.fc_self(x) + self.fc_neigh(neigh)
        return self.norm(self.dropout(F.relu(out)))


class DVEB(torch.nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.graphsage = WeightedGraphSAGE(hidden_dim, dropout)

    def forward(self, x: torch.Tensor, weighted_adj: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        return self.graphsage(x, weighted_adj)


class DVTT(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        hidden_dim: int,
        mlp_dim: int,
        num_classes: int,
        num_dveb: int = 3,
        nhead: int = 4,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.node_encoder = torch.nn.Linear(feature_dim, hidden_dim)
        self.topology_encoder = torch.nn.Linear(num_nodes, hidden_dim)
        self.sequence_position = torch.nn.Parameter(torch.empty(num_nodes, hidden_dim))
        torch.nn.init.xavier_uniform_(self.sequence_position)

        self.blocks = torch.nn.ModuleList([
            DVEB(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(num_dveb)
        ])

        self.soft_attention = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Linear(hidden_dim, mlp_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(mlp_dim, num_classes)

    def forward(self, node_features: torch.Tensor, weighted_adj: torch.Tensor) -> torch.Tensor:
        topology_encoding = self.topology_encoder(weighted_adj)
        x = self.node_encoder(node_features) + topology_encoding + self.sequence_position

        for block in self.blocks:
            x = block(x, weighted_adj)

        gate = torch.sigmoid(self.soft_attention(x))
        x = self.norm(gate * x)
        x = F.gelu(self.mlp(x))

        # The paper combines sum- and max-pooling views before the FC layer.
        # Keeping a singleton view dimension preserves node-wise classification.
        views = x.unsqueeze(1)
        pooled = torch.sum(views, dim=1) + torch.amax(views, dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
