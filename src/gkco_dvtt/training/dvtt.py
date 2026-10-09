import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def _accuracy(
    logits,
    labels,
    mask,
):

    prediction = (
        logits[mask]
        .max(1)[1]
    )

    correct = (
        prediction
        .eq(labels[mask])
        .sum()
        .item()
    )

    total = (
        mask.sum().item()
    )

    if total == 0:
        return 0.0

    return correct / total


def train_dvtt(
    model,
    features,
    labels,
    adjacency,
    weights,
    train_mask,
    val_mask,
    test_mask,
    learning_rate,
    epochs,
    device,
    log_every=50,
):

    model = model.to(device)

    features = features.to(device)
    labels = labels.to(device)

    adjacency = adjacency.to(device)
    weights = weights.to(device)

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_validation_accuracy = -1.0
    best_test_accuracy = 0.0
    best_test_f1 = 0.0
    best_epoch = 0

    best_state = None

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            features,
            adjacency,
            weights,
        )

        loss = F.nll_loss(
            logits[train_mask],
            labels[train_mask],
        )

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            logits = model(
                features,
                adjacency,
                weights,
            )

            validation_accuracy = (
                _accuracy(
                    logits,
                    labels,
                    val_mask,
                )
            )

            test_accuracy = (
                _accuracy(
                    logits,
                    labels,
                    test_mask,
                )
            )

            test_prediction = (
                logits[test_mask]
                .max(1)[1]
                .cpu()
                .numpy()
            )

            test_labels = (
                labels[test_mask]
                .cpu()
                .numpy()
            )

            test_f1 = f1_score(
                test_labels,
                test_prediction,
                average="weighted",
            )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            best_test_accuracy = (
                test_accuracy
            )

            best_test_f1 = (
                test_f1
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

        if (
            epoch == 1
            or epoch % log_every == 0
            or epoch == epochs
        ):

            print(
                "[DVTT] "
                "epoch "
                "{:04d}/{:04d}  "
                "loss={:.6f}  "
                "val_acc={:.4f}".format(
                    epoch,
                    epochs,
                    loss.item(),
                    validation_accuracy,
                )
            )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return {
        "best_validation_accuracy":
            float(
                best_validation_accuracy
            ),

        "test_accuracy":
            float(
                best_test_accuracy
            ),

        "test_f1_weighted":
            float(
                best_test_f1
            ),

        "best_epoch":
            int(
                best_epoch
            ),
    }