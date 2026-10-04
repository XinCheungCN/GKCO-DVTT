import torch
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, SAGEConv
from torch_geometric.utils import get_laplacian, to_dense_adj


class Encoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.01):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv_mu = SAGEConv(hidden_dim, latent_dim)
        self.conv_logvar = SAGEConv(hidden_dim, latent_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm_mu = LayerNorm(latent_dim)
        self.norm_logvar = LayerNorm(latent_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.norm1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        mu = self.norm_mu(self.conv_mu(h, edge_index))
        logvar = torch.clamp(self.norm_logvar(self.conv_logvar(h, edge_index)), -5.0, 5.0)
        return mu, logvar


class LatentLayer(torch.nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc_mu = torch.nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(latent_dim, latent_dim)
        self.sage_mu = SAGEConv(latent_dim, latent_dim)
        self.sage_logvar = SAGEConv(latent_dim, latent_dim)
        self.norm_mu = LayerNorm(latent_dim)
        self.norm_logvar = LayerNorm(latent_dim)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.norm_mu(self.sage_mu(self.fc_mu(mu), edge_index))
        logvar = self.norm_logvar(self.sage_logvar(self.fc_logvar(logvar), edge_index))
        return mu, torch.clamp(logvar, -5.0, 5.0)


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = SAGEConv(latent_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(z, edge_index)))
        h = self.dropout(h)
        return self.norm2(self.conv2(h, edge_index))


class GCVA(torch.nn.Module):
    def __init__(self, in_dim: int, encoder_hidden: int, latent_dim: int, decoder_hidden: int, dropout: float = 0.01):
        super().__init__()
        self.encoder = Encoder(in_dim, encoder_hidden, latent_dim, dropout)
        self.latent = LatentLayer(latent_dim)
        self.decoder = Decoder(latent_dim, decoder_hidden, in_dim, dropout)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, edge_index)
        mu, logvar = self.latent(mu, logvar, edge_index)
        z = self.reparameterize(mu, logvar, self.training)
        return z, mu, logvar

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x, edge_index)
        recon = self.decoder(z, edge_index)
        return recon, z, mu, logvar


def laplacian_regularization(mu: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    lap_edge, lap_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    lap = to_dense_adj(lap_edge, edge_attr=lap_weight, max_num_nodes=num_nodes).squeeze(0)
    lap = lap.to(mu.device, dtype=mu.dtype)
    return torch.trace(mu.T @ lap @ mu)
