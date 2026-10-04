from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from ..models.dvtt import DVTT


@dataclass
class DVTTResult:
    best_val_accuracy: float
    test_accuracy: float
    test_f1_weighted: float
    best_epoch: int


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=1)
    return float((pred == labels[mask]).float().mean().item())


def train_dvtt(
    model: DVTT,
    node_features: torch.Tensor,
    weighted_adj: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    lr: float,
    epochs: int,
    verbose: bool = True,
) -> DVTTResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(node_features, weighted_adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(node_features, weighted_adj)
            val_acc = _accuracy(logits, labels, val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print(f"[DVTT] epoch {epoch:04d}/{epochs}  loss={float(loss):.6f}  val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(node_features, weighted_adj)
        pred = logits[test_mask].argmax(dim=1).cpu().numpy()
        true = labels[test_mask].cpu().numpy()

    return DVTTResult(
        best_val_accuracy=best_val,
        test_accuracy=float(accuracy_score(true, pred)),
        test_f1_weighted=float(f1_score(true, pred, average="weighted")),
        best_epoch=best_epoch,
    )
