import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGATLayer(nn.Module):
    """
    Dense Graph Attention Network (GAT) layer.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output dimension of each attention head.
    heads : int
        Number of attention heads.
    concat : bool
        Whether to concatenate outputs from different heads.
    dropout : float
        Dropout probability.
    negative_slope : float
        Negative slope used in LeakyReLU.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        heads=4,
        concat=True,
        dropout=0.0,
        negative_slope=0.2,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Feature transformation
        self.lin = nn.Linear(
            in_dim,
            heads * out_dim,
            bias=False
        )

        # Attention parameters
        self.att_l = nn.Parameter(
            torch.empty(heads, out_dim)
        )
        self.att_r = nn.Parameter(
            torch.empty(heads, out_dim)
        )

        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

        output_dim = heads * out_dim if concat else out_dim
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, adj):
        """
        Parameters
        ----------
        x : Tensor
            Node features with shape [B, N, F].
        adj : Tensor
            Adjacency matrix with shape [B, N, N].

        Returns
        -------
        Tensor
            Updated node features.
        """

        batch_size, num_nodes, _ = x.shape

        # --------------------------------------------------
        # 1. Linear projection
        # [B, N, F]
        #     ->
        # [B, N, H, F_out]
        # --------------------------------------------------
        h = self.lin(x).view(
            batch_size,
            num_nodes,
            self.heads,
            self.out_dim,
        )

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training
        )

        # --------------------------------------------------
        # 2. Compute attention scores
        # --------------------------------------------------
        att_left = (
            h * self.att_l[None, None, :, :]
        ).sum(dim=-1)

        att_right = (
            h * self.att_r[None, None, :, :]
        ).sum(dim=-1)

        attention = (
            att_left.permute(0, 2, 1).unsqueeze(-1)
            + att_right.permute(0, 2, 1).unsqueeze(-2)
        )

        attention = F.leaky_relu(
            attention,
            negative_slope=self.negative_slope,
        )

        # Only retain connected node pairs
        mask = (adj > 0).unsqueeze(1)
        attention = attention.masked_fill(~mask, -1e9)

        # --------------------------------------------------
        # 3. Normalize attention coefficients
        # --------------------------------------------------
        alpha = F.softmax(attention, dim=-1)

        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training
        )

        # --------------------------------------------------
        # 4. Aggregate neighboring node features
        # --------------------------------------------------
        h = h.permute(0, 2, 1, 3)

        out = torch.matmul(alpha, h)

        out = out.permute(
            0, 2, 1, 3
        ).contiguous()

        if self.concat:
            out = out.view(
                batch_size,
                num_nodes,
                self.heads * self.out_dim,
            )
        else:
            out = out.mean(dim=2)

        return out + self.bias


class GATGraphRegressor(nn.Module):
    """
    Two-layer GAT for graph-level regression.

    Architecture
    ------------
    Node features
        ↓
    Multi-head GAT
        ↓
    GAT
        ↓
    Global mean pooling
        ↓
    MLP
        ↓
    Graph-level scalar
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=256,
        heads=4,
        dropout=0.0,
    ):
        super().__init__()

        self.dropout = dropout

        # GAT layer 1:
        # in_dim -> hidden_dim × heads
        self.gat1 = DenseGATLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        # GAT layer 2:
        # hidden_dim × heads -> 64
        self.gat2 = DenseGATLayer(
            in_dim=hidden_dim * heads,
            out_dim=64,
            heads=1,
            concat=True,
            dropout=dropout,
        )

        # Graph-level regression head
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, adj, mask):
        """
        Parameters
        ----------
        x : Tensor
            Node features: [B, N, F]

        adj : Tensor
            Adjacency matrix: [B, N, N]

        mask : Tensor
            Node mask: [B, N]

        Returns
        -------
        Tensor
            Graph-level prediction: [B]
        """

        # GAT layer 1
        h = F.elu(self.gat1(x, adj))

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )

        # GAT layer 2
        h = F.elu(self.gat2(h, adj))

        # -----------------------------------------------
        # Global masked mean pooling
        # -----------------------------------------------
        mask = mask.unsqueeze(-1)

        h = h * mask

        graph_feature = (
            h.sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )

        # Graph-level prediction
        return self.mlp(graph_feature).squeeze(-1)


if __name__ == "__main__":

    # Example:
    # node features = [node features + global features]
    input_dim = 7

    model = GATGraphRegressor(
        in_dim=input_dim,
        hidden_dim=256,
        heads=4,
        dropout=0.0,
    )

    print(model)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")