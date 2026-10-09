import torch
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        dropout=0.0,
    ):
        super().__init__()

        self.fc_self = torch.nn.Linear(
            in_feats,
            out_feats,
            bias=False,
        )

        self.fc_neigh = torch.nn.Linear(
            in_feats,
            out_feats,
            bias=False,
        )

        self.dropout = torch.nn.Dropout(
            dropout
        )

        self.layer_norm = torch.nn.LayerNorm(
            out_feats
        )

    def forward(
        self,
        x,
        adj,
    ):
        """
        x:
            [N, F]

        adj:
            [N, N]
        """

        adj = adj.float()

        degree = adj.sum(
            dim=1,
            keepdim=True,
        )

        degree = torch.clamp(
            degree,
            min=1e-12,
        )

        neighbor_features = (
            torch.matmul(
                adj,
                x,
            )
            / degree
        )

        self_features = (
            self.fc_self(x)
        )

        neighbor_features = (
            self.fc_neigh(
                neighbor_features
            )
        )

        out = (
            self_features
            + neighbor_features
        )

        out = F.relu(out)

        out = self.dropout(out)

        out = self.layer_norm(out)

        return out


class DVTT(torch.nn.Module):

    def __init__(
        self,
        feature_dim_size,
        num_nodes,
        hidden_size,
        hidden_size2,
        num_classes,
        num_self_att_layers=1,
        num_gnn_layers=3,
        nhead=4,
        dropout=0.01,
    ):
        super().__init__()

        self.num_gnn_layers = (
            num_gnn_layers
        )

        # Layer 1: dual encoding

        self.feature_embedding = (
            torch.nn.Linear(
                feature_dim_size,
                hidden_size,
            )
        )

        # Topology encoding f_t(A_i)
        self.topology_embedding = (
            torch.nn.Linear(
                num_nodes,
                hidden_size,
            )
        )

        # Learnable sequence-position encoding
        self.position_embedding = (
            torch.nn.Parameter(
                torch.zeros(
                    num_nodes,
                    hidden_size,
                )
            )
        )

        torch.nn.init.xavier_uniform_(
            self.position_embedding
        )

        # Layer 2: DVEBs

        self.transformer_layers = (
            torch.nn.ModuleList()
        )

        self.graphsage_layers = (
            torch.nn.ModuleList()
        )

        for _ in range(
            num_gnn_layers
        ):

            encoder_layer = (
                torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=nhead,
                    dim_feedforward=hidden_size,
                    dropout=dropout,
                    batch_first=True,
                )
            )

            transformer = (
                torch.nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=(
                        num_self_att_layers
                    ),
                )
            )

            self.transformer_layers.append(
                transformer
            )

            self.graphsage_layers.append(
                GraphSAGE(
                    hidden_size,
                    hidden_size,
                    dropout=dropout,
                )
            )

        # Layer 3: soft attention + MLP

        self.soft_attention = (
            torch.nn.Linear(
                hidden_size,
                1,
            )
        )

        self.layer_norm = (
            torch.nn.LayerNorm(
                hidden_size
            )
        )

        self.mlp = torch.nn.Linear(
            hidden_size,
            hidden_size2,
        )

        self.dropout = torch.nn.Dropout(
            dropout
        )

        self.classifier = torch.nn.Linear(
            hidden_size2,
            num_classes,
        )

    def forward(
        self,
        inputs,
        adj,
        weights,
    ):

        # Layer 1

        node_features = (
            self.feature_embedding(
                inputs
            )
        )

        topology_features = (
            self.topology_embedding(
                weights.float()
            )
        )

        x = (
            node_features
            + topology_features
            + self.position_embedding
        )

        # Layer 2: three DVEBs

        for i in range(
            self.num_gnn_layers
        ):

            # Transformer expects:
            # [batch, sequence, feature]
            transformer_input = (
                x.unsqueeze(0)
            )

            transformer_output = (
                self.transformer_layers[i](
                    transformer_input
                )
            )

            x = transformer_output.squeeze(
                0
            )

            x = self.graphsage_layers[i](
                x,
                adj,
            )

        # Layer 3: soft attention

        attention = torch.sigmoid(
            self.soft_attention(x)
        )

        x = self.layer_norm(
            attention * x
        )

        x = self.mlp(x)

        x = F.gelu(x)

        # Dual pooling

        pooling_input = x.unsqueeze(1)

        sum_pooling = torch.sum(
            pooling_input,
            dim=1,
        )

        max_pooling = torch.amax(
            pooling_input,
            dim=1,
        )

        graph_embeddings = (
            sum_pooling
            + max_pooling
        )

        graph_embeddings = self.dropout(
            graph_embeddings
        )

        logits = self.classifier(
            graph_embeddings
        )

        return F.log_softmax(
            logits,
            dim=-1,
        )