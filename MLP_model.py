import torch
import torch.nn as nn


# ============================================================
# Input configuration
# ============================================================

MAX_TURBINES = 80
NODE_FEATURE_DIM = 2       # x, y
MASK_DIM = MAX_TURBINES    # turbine existence mask
GLOBAL_FEATURE_DIM = 3     # wind speed, wind direction, TI

INPUT_DIM = (
    MAX_TURBINES * NODE_FEATURE_DIM
    + MASK_DIM
    + GLOBAL_FEATURE_DIM
)

# 80 × 2 + 80 + 3 = 243


# ============================================================
# MLP Regression Model
# ============================================================

class MLPRegressor(nn.Module):
    """
    MLP for wind-farm-level regression.

    Input
    -----
    243-dimensional vector:

        160 turbine-coordinate features:
            80 turbines × (x, y)

        80 turbine-existence mask values

        3 global inflow features:
            wind speed,
            wind direction,
            turbulence intensity

    Output
    ------
    Scalar graph/layout-level prediction.
    """

    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":

    model = MLPRegressor()

    print(model)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nInput dimension: {INPUT_DIM}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Example batch
    x = torch.randn(4, INPUT_DIM)

    model.eval()
    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)