import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import DatasetBundle, build_masks, load_numpy_dataset
from .graph import build_graph_triplet, build_weighted_optimized_graph, half_spectrum
from .models.dvtt import DVTT
from .models.gcva import GCVA
from .preprocessing.noise import add_gaussian_noise_by_snr
from .training.dvtt import train_dvtt
from .training.gkco import optimize_node_features, train_gcva


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_optional_noise(bundle: DatasetBundle, cfg: dict) -> DatasetBundle:
    noise_cfg = cfg.get("preprocessing", {}).get("noise", {})
    if not noise_cfg.get("enabled", False):
        return bundle
    snr_db = float(noise_cfg["snr_db"])
    seed = int(noise_cfg.get("seed", cfg.get("seed", 42)))
    return DatasetBundle(
        train_x=add_gaussian_noise_by_snr(bundle.train_x, snr_db, seed),
        train_y=bundle.train_y,
        val_x=add_gaussian_noise_by_snr(bundle.val_x, snr_db, seed + 1),
        val_y=bundle.val_y,
        test_x=add_gaussian_noise_by_snr(bundle.test_x, snr_db, seed + 2),
        test_y=bundle.test_y,
    )


def run_pipeline(cfg: dict) -> dict:
    _set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    bundle = load_numpy_dataset(cfg["dataset"]["root"])
    bundle = _apply_optional_noise(bundle, cfg)
    labels_np = bundle.all_y.astype(np.int64)
    features_np = half_spectrum(bundle.all_x)

    graph_cfg = cfg["graph"]
    graphs = build_graph_triplet(
        features_np,
        k_original=int(graph_cfg["k_original"]),
        k_positive=int(graph_cfg.get("k_positive", 1)),
        k_negative=int(graph_cfg.get("k_negative", 1)),
    )
    graphs.original = graphs.original.to(device)
    graphs.positive = graphs.positive.to(device)
    graphs.negative = graphs.negative.to(device)

    g_cfg = cfg["gcva"]
    gcva = GCVA(
        in_dim=features_np.shape[1],
        encoder_hidden=int(g_cfg["encoder_hidden"]),
        latent_dim=int(g_cfg["latent_dim"]),
        decoder_hidden=int(g_cfg["decoder_hidden"]),
        dropout=float(g_cfg.get("dropout", 0.01)),
    ).to(device)
    gkco_history = train_gcva(
        gcva,
        graphs,
        lr=float(g_cfg["learning_rate"]),
        epochs=int(g_cfg["epochs"]),
        margin=float(g_cfg.get("contrastive_margin", 1.0)),
        verbose=bool(cfg.get("verbose", True)),
    )
    optimized_features = optimize_node_features(gcva, graphs.original)
    _, weighted_adj = build_weighted_optimized_graph(optimized_features, int(graph_cfg["k_optimized"]))

    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    train_mask, val_mask, test_mask = build_masks(bundle.split_sizes, device)

    d_cfg = cfg["dvtt"]
    dvtt = DVTT(
        feature_dim=optimized_features.shape[1],
        num_nodes=optimized_features.shape[0],
        hidden_dim=int(d_cfg["hidden_dim"]),
        mlp_dim=int(d_cfg["mlp_dim"]),
        num_classes=int(cfg["dataset"]["num_classes"]),
        num_dveb=int(d_cfg.get("num_dveb", 3)),
        nhead=int(d_cfg.get("nhead", 4)),
        dropout=float(d_cfg.get("dropout", 0.01)),
    ).to(device)

    result = train_dvtt(
        dvtt,
        optimized_features,
        weighted_adj,
        labels,
        train_mask,
        val_mask,
        test_mask,
        lr=float(d_cfg["learning_rate"]),
        epochs=int(d_cfg["epochs"]),
        verbose=bool(cfg.get("verbose", True)),
    )

    out_dir = Path(cfg.get("output_dir", "outputs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gcva.state_dict(), out_dir / "gcva.pt")
    torch.save(dvtt.state_dict(), out_dir / "dvtt.pt")
    np.save(out_dir / "optimized_features.npy", optimized_features.detach().cpu().numpy())
    np.save(out_dir / "weighted_adjacency.npy", weighted_adj.detach().cpu().numpy())
    np.save(out_dir / "gkco_loss.npy", np.asarray(gkco_history, dtype=np.float32))

    metrics = {
        "best_validation_accuracy": result.best_val_accuracy,
        "test_accuracy": result.test_accuracy,
        "test_f1_weighted": result.test_f1_weighted,
        "best_epoch": result.best_epoch,
        "device": str(device),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
