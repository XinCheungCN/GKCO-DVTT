from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm
from torch_geometric.utils import get_laplacian, to_dense_adj, to_undirected


class Encoder(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.01,
    ):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)

        self.conv_mu = SAGEConv(
            hidden_dim,
            latent_dim,
        )

        self.conv_logvar = SAGEConv(
            hidden_dim,
            latent_dim,
        )

        self.norm1 = LayerNorm(hidden_dim)
        self.norm_mu = LayerNorm(latent_dim)
        self.norm_logvar = LayerNorm(latent_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.dropout(x)

        mu = self.conv_mu(x, edge_index)
        mu = self.norm_mu(mu)

        logvar = self.conv_logvar(
            x,
            edge_index,
        )

        logvar = self.norm_logvar(logvar)
        logvar = torch.clamp(
            logvar,
            -5.0,
            5.0,
        )

        return mu, logvar


class LatentLayer(torch.nn.Module):

    def __init__(
        self,
        latent_dim: int,
        edge_index_for_laplacian: torch.Tensor,
        num_nodes: int,
        lambda_reg: float = 1e-3,
    ):
        super().__init__()

        self.fc_mu = torch.nn.Linear(
            latent_dim,
            latent_dim,
        )

        self.fc_logvar = torch.nn.Linear(
            latent_dim,
            latent_dim,
        )

        self.sage_mu = SAGEConv(
            latent_dim,
            latent_dim,
        )

        self.sage_logvar = SAGEConv(
            latent_dim,
            latent_dim,
        )

        self.bn_mu = torch.nn.BatchNorm1d(
            latent_dim
        )

        self.bn_logvar = torch.nn.BatchNorm1d(
            latent_dim
        )

        self.norm_mu = LayerNorm(
            latent_dim
        )

        self.norm_logvar = LayerNorm(
            latent_dim
        )

        self.lambda_reg = lambda_reg

        # Symmetrize the graph for Laplacian regularization.
        laplacian_edges = to_undirected(
            edge_index_for_laplacian,
            num_nodes=num_nodes,
        )
        laplacian_edge_index, laplacian_weight = get_laplacian(
            laplacian_edges,
            normalization="sym",
            num_nodes=num_nodes,
        )

        laplacian_matrix = to_dense_adj(
            laplacian_edge_index,
            edge_attr=laplacian_weight,
            max_num_nodes=num_nodes,
        ).squeeze(0)

        self.register_buffer(
            "laplacian_matrix",
            laplacian_matrix,
        )

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        edge_index: torch.Tensor,
    ):

        mu = self.fc_mu(mu)
        logvar = self.fc_logvar(logvar)

        mu = self.norm_mu(mu)
        logvar = self.norm_logvar(logvar)

        mu = self.sage_mu(
            mu,
            edge_index,
        )

        logvar = self.sage_logvar(
            logvar,
            edge_index,
        )

        mu = self.bn_mu(mu)
        logvar = self.bn_logvar(logvar)

        laplacian_loss = torch.trace(
            mu.T
            @ self.laplacian_matrix
            @ mu
        )

        laplacian_loss = (
            self.lambda_reg
            * laplacian_loss
        )

        return (
            mu,
            logvar,
            laplacian_loss,
        )


class Decoder(torch.nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            latent_dim,
            hidden_dim,
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            output_dim,
        )

        self.norm1 = LayerNorm(
            hidden_dim
        )

        self.norm2 = LayerNorm(
            output_dim
        )

        self.dropout = torch.nn.Dropout(
            dropout
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ):

        z = self.conv1(
            z,
            edge_index,
        )

        z = self.norm1(z)
        z = F.relu(z)

        z = self.dropout(z)

        z = self.conv2(
            z,
            edge_index,
        )

        z = self.norm2(z)

        return z


class GCVA(torch.nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        latent_layer: LatentLayer,
        decoder: Decoder,
    ):
        super().__init__()

        self.encoder = encoder
        self.latent_layer = latent_layer
        self.decoder = decoder

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):

        if self.training:
            noise = torch.randn_like(
                logvar
            )

            return (
                mu
                + noise
                * torch.exp(
                    0.5 * logvar
                )
            )

        return mu

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):

        mu, logvar = self.encoder(
            x,
            edge_index,
        )

        (
            mu,
            logvar,
            laplacian_loss,
        ) = self.latent_layer(
            mu,
            logvar,
            edge_index,
        )

        z = self.reparameterize(
            mu,
            logvar,
        )

        recon_x = self.decoder(
            z,
            edge_index,
        )

        return (
            recon_x,
            mu,
            logvar,
            laplacian_loss,
        )