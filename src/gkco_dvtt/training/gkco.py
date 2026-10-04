from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..graph import GraphTriplet
from ..models.gcva import GCVA, laplacian_regularization


@dataclass
class LossTerms:
    reconstruction: torch.Tensor
    kl: torch.Tensor
    laplacian: torch.Tensor
    contrastive: torch.Tensor


def _contrastive_loss(z: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor, margin: float) -> torch.Tensor:
    pos_sim = F.cosine_similarity(z, z_pos, dim=1)
    neg_sim = F.cosine_similarity(z, z_neg, dim=1)
    return F.relu(pos_sim - neg_sim + margin).mean()


def _loss_terms(model: GCVA, graphs: GraphTriplet, margin: float) -> tuple[LossTerms, torch.Tensor]:
    recon, z, mu, logvar = model(graphs.original.x, graphs.original.edge_index)
    _, z_pos, _, _ = model(graphs.positive.x, graphs.positive.edge_index)
    _, z_neg, _, _ = model(graphs.negative.x, graphs.negative.edge_index)

    reconstruction = F.mse_loss(recon, graphs.original.x, reduction="sum")
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    laplacian = laplacian_regularization(mu, graphs.original.edge_index, graphs.original.num_nodes)
    contrastive = _contrastive_loss(z, z_pos, z_neg, margin)
    return LossTerms(reconstruction, kl, laplacian, contrastive), recon


def _normalizers(terms: LossTerms) -> dict[str, torch.Tensor]:
    eps = 1e-12
    return {
        "reconstruction": terms.reconstruction.detach().abs().clamp_min(eps),
        "kl": terms.kl.detach().abs().clamp_min(eps),
        "laplacian": terms.laplacian.detach().abs().clamp_min(eps),
        "contrastive": terms.contrastive.detach().abs().clamp_min(eps),
    }


def train_gcva(model: GCVA, graphs: GraphTriplet, lr: float, epochs: int, margin: float, verbose: bool = True) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    initial_terms, _ = _loss_terms(model, graphs, margin)
    norm = _normalizers(initial_terms)
    history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        terms, _ = _loss_terms(model, graphs, margin)
        loss = (
            terms.reconstruction / norm["reconstruction"]
            + terms.kl / norm["kl"]
            + terms.laplacian / norm["laplacian"]
            + terms.contrastive / norm["contrastive"]
        )
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append(value)
        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            print(f"[GKCO] epoch {epoch:04d}/{epochs}  loss={value:.6f}")
    return history


@torch.no_grad()
def optimize_node_features(model: GCVA, original_graph) -> torch.Tensor:
    model.eval()
    recon, _, _, _ = model(original_graph.x, original_graph.edge_index)
    return recon
