import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .graph import (
    build_gkco_graphs,
    build_rank_weighted_graph,
    fft_half_spectrum,
)

from .models.gcva import (
    Decoder,
    Encoder,
    GCVA,
    LatentLayer,
)

from .models.dvtt import DVTT

from .training.gkco import (
    reconstruct_features,
    train_gkco,
)

from .training.dvtt import (
    train_dvtt,
)


def _set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def _load_array(
    root,
    filename,
):
    path = root / filename

    if not path.exists():
        raise FileNotFoundError(
            "Dataset file not found: "
            + str(path)
        )

    return np.load(path)


def _flatten_samples(x):

    x = np.asarray(x)

    if x.ndim > 2:
        x = x.reshape(
            x.shape[0],
            -1,
        )

    return x

def _build_masks(
    train_size,
    validation_size,
    test_size,
):

    total_size = (
        train_size
        + validation_size
        + test_size
    )

    train_mask = torch.zeros(
        total_size,
        dtype=torch.bool,
    )

    val_mask = torch.zeros(
        total_size,
        dtype=torch.bool,
    )

    test_mask = torch.zeros(
        total_size,
        dtype=torch.bool,
    )

    train_mask[
        :train_size
    ] = True

    val_start = train_size
    val_end = (
        train_size
        + validation_size
    )

    val_mask[
        val_start:val_end
    ] = True

    test_mask[
        val_end:
    ] = True

    return (
        train_mask,
        val_mask,
        test_mask,
    )


def run_pipeline(config):

    experiment_config = (
        config.get(
            "experiment",
            {},
        )
    )

    seed = experiment_config.get(
        "seed",
        42,
    )

    _set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )

    dataset_config = (
        config["dataset"]
    )

    data_root = Path(
        dataset_config["root"]
    )

    num_classes = int(
        dataset_config[
            "num_classes"
        ]
    )

    train_x = _flatten_samples(
        _load_array(
            data_root,
            "train_x.npy",
        )
    )

    train_y = _load_array(
        data_root,
        "train_y.npy",
    )

    validation_x = (
        _flatten_samples(
            _load_array(
                data_root,
                "validation_x.npy",
            )
        )
    )

    validation_y = _load_array(
        data_root,
        "validation_y.npy",
    )

    test_x = _flatten_samples(
        _load_array(
            data_root,
            "test_x.npy",
        )
    )

    test_y = _load_array(
        data_root,
        "test_y.npy",
    )

    # Build one graph from the supplied data arrays.

    raw_samples = np.concatenate(
        [
            train_x,
            validation_x,
            test_x,
        ],
        axis=0,
    )

    labels = np.concatenate(
        [
            train_y,
            validation_y,
            test_y,
        ],
        axis=0,
    )

    labels = labels.astype(
        np.int64
    )

    train_mask, val_mask, test_mask = (
        _build_masks(
            len(train_x),
            len(validation_x),
            len(test_x),
        )
    )

    print(
        "Dataset sizes:"
    )

    print(
        "train = {}, validation = {}, test = {}".format(
            len(train_x),
            len(validation_x),
            len(test_x),
        )
    )

    # FFT node construction

    node_features = (
        fft_half_spectrum(
            raw_samples
        )
    )

    feature_dim = (
        node_features.shape[1]
    )

    graph_config = (
        config["graph"]
    )

    (
        edge_original,
        edge_positive,
        edge_negative,
    ) = build_gkco_graphs(

        node_features,

        k_original=int(
            graph_config[
                "k_original"
            ]
        ),

        k_positive=int(
            graph_config.get(
                "k_positive",
                1,
            )
        ),

        k_negative=int(
            graph_config.get(
                "k_negative",
                1,
            )
        ),
    )

    node_tensor = torch.tensor(
        node_features,
        dtype=torch.float32,
    )

    data_original = Data(
        x=node_tensor,
        edge_index=edge_original,
    )

    data_positive = Data(
        x=node_tensor,
        edge_index=edge_positive,
    )

    data_negative = Data(
        x=node_tensor,
        edge_index=edge_negative,
    )

    # GCVA

    gcva_config = (
        config["gcva"]
    )

    hidden_dim = int(
        gcva_config[
            "hidden_dim"
        ]
    )

    latent_dim = int(
        gcva_config[
            "latent_dim"
        ]
    )

    decoder_hidden = int(
        gcva_config.get(
            "decoder_hidden",
            hidden_dim,
        )
    )

    encoder = Encoder(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=float(
            gcva_config.get(
                "dropout",
                0.01,
            )
        ),
    )

    # Apply the Laplacian penalty to the original graph.
    latent_layer = LatentLayer(
        latent_dim=latent_dim,
        edge_index_for_laplacian=(
            edge_original
        ),
        num_nodes=(
            node_features.shape[0]
        ),
        lambda_reg=float(
            gcva_config.get(
                "lambda_reg",
                1e-3,
            )
        ),
    )

    decoder = Decoder(
        latent_dim=latent_dim,
        hidden_dim=decoder_hidden,
        output_dim=feature_dim,
        dropout=float(
            gcva_config.get(
                "decoder_dropout",
                0.0,
            )
        ),
    )

    gcva = GCVA(
        encoder,
        latent_layer,
        decoder,
    ).to(device)

    gcva_optimizer = (
        torch.optim.Adam(
            gcva.parameters(),
            lr=float(
                gcva_config[
                    "learning_rate"
                ]
            ),
        )
    )

    train_gkco(
        model=gcva,
        data_original=data_original,
        data_positive=data_positive,
        data_negative=data_negative,
        optimizer=gcva_optimizer,
        epochs=int(
            gcva_config[
                "epochs"
            ]
        ),
        device=device,
        beta=float(
            gcva_config.get(
                "beta",
                1.0,
            )
        ),
    )

    decoded_features = (
        reconstruct_features(
            gcva,
            data_original,
            device,
        )
    )

    decoded_features_cpu = (
        decoded_features
        .detach()
        .cpu()
    )

    # Downstream graph

    k_optimized = int(
        graph_config.get(
            "k_optimized",
            5,
        )
    )

    adjacency, weights = (
        build_rank_weighted_graph(
            decoded_features_cpu.numpy(),
            k=k_optimized,
        )
    )

    # DVTT

    dvtt_config = (
        config["dvtt"]
    )

    model = DVTT(

        feature_dim_size=(
            decoded_features_cpu.shape[1]
        ),

        num_nodes=(
            decoded_features_cpu.shape[0]
        ),

        hidden_size=int(
            dvtt_config[
                "hidden_dim"
            ]
        ),

        hidden_size2=int(
            dvtt_config[
                "mlp_hidden_dim"
            ]
        ),

        num_classes=num_classes,

        num_self_att_layers=int(
            dvtt_config.get(
                "num_self_att_layers",
                2,
            )
        ),

        num_gnn_layers=int(
            dvtt_config.get(
                "num_gnn_layers",
                2,
            )
        ),

        nhead=int(
            dvtt_config.get(
                "nhead",
                4,
            )
        ),

        dropout=float(
            dvtt_config.get(
                "dropout",
                0.01,
            )
        ),
    )

    label_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    metrics = train_dvtt(

        model=model,

        features=(
            decoded_features_cpu
        ),

        labels=label_tensor,

        adjacency=adjacency,

        weights=weights,

        train_mask=train_mask,

        val_mask=val_mask,

        test_mask=test_mask,

        learning_rate=float(
            dvtt_config[
                "learning_rate"
            ]
        ),

        epochs=int(
            dvtt_config[
                "epochs"
            ]
        ),

        device=device,
    )

    metrics["device"] = str(
        device
    )

    # Save

    experiment_name = (
        experiment_config.get(
            "name",
            "default",
        )
    )

    output_dir = (
        Path("outputs")
        / experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir
        / "optimized_features.npy",

        decoded_features_cpu.numpy(),
    )

    np.save(
        output_dir
        / "labels.npy",
        labels,
    )

    torch.save(
        model.state_dict(),
        output_dir
        / "dvtt_best.pt",
    )

    with (
        output_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    print()
    print("Final metrics")

    for key, value in (
        metrics.items()
    ):
        print(
            "{}: {}".format(
                key,
                value,
            )
        )

    return metrics