import torch
import torch.nn.functional as F


def gcva_loss(
    recon_x,
    x,
    mu,
    logvar,
    laplacian_loss,
    beta=1.0,
):
    """Compute reconstruction, KL, and Laplacian losses."""

    reconstruction_loss = F.mse_loss(
        recon_x,
        x,
        reduction="sum",
    )

    kl_loss = -0.5 * torch.sum(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    total_loss = (
        reconstruction_loss
        + beta * kl_loss
        + laplacian_loss
    )

    return total_loss


def contrastive_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
):

    positive_similarity = (
        F.cosine_similarity(
            anchor,
            positive,
        )
    )

    negative_similarity = (
        F.cosine_similarity(
            anchor,
            negative,
        )
    )

    loss = F.relu(
        positive_similarity
        - negative_similarity
        + margin
    )

    return loss.mean()


def train_gkco(
    model,
    data_original,
    data_positive,
    data_negative,
    optimizer,
    epochs,
    device,
    beta=1.0,
    log_every=25,
):

    data_original = data_original.to(
        device
    )

    data_positive = data_positive.to(
        device
    )

    data_negative = data_negative.to(
        device
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()
        optimizer.zero_grad()

        (
            recon_original,
            mu_original,
            logvar_original,
            laplacian_original,
        ) = model(
            data_original.x,
            data_original.edge_index,
        )

        (
            _,
            mu_positive,
            _,
            _,
        ) = model(
            data_positive.x,
            data_positive.edge_index,
        )

        (
            _,
            mu_negative,
            _,
            _,
        ) = model(
            data_negative.x,
            data_negative.edge_index,
        )

        loss_original = gcva_loss(
            recon_original,
            data_original.x,
            mu_original,
            logvar_original,
            laplacian_original,
            beta=beta,
        )

        loss_contrastive = (
            contrastive_loss(
                mu_original,
                mu_positive,
                mu_negative,
            )
        )

        total_loss = (
            loss_original
            + loss_contrastive
        )

        total_loss.backward()
        optimizer.step()

        if (
            epoch == 1
            or epoch % log_every == 0
            or epoch == epochs
        ):
            print(
                "[GKCO] "
                "epoch "
                "{:04d}/{:04d}  "
                "loss={:.6f}".format(
                    epoch,
                    epochs,
                    total_loss.item(),
                )
            )


@torch.no_grad()
def reconstruct_features(
    model,
    data_original,
    device,
):

    model.eval()

    data_original = (
        data_original.to(device)
    )

    mu, logvar = model.encoder(
        data_original.x,
        data_original.edge_index,
    )

    (
        mu,
        logvar,
        _,
    ) = model.latent_layer(
        mu,
        logvar,
        data_original.edge_index,
    )

    z = model.reparameterize(
        mu,
        logvar,
    )

    decoded_features = model.decoder(
        z,
        data_original.edge_index,
    )

    return decoded_features