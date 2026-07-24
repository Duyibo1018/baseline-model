import torch
import torch.nn as nn


class CNNRegressor(nn.Module):
    """
    CNN for graph/layout-level regression.

    Input:
        4-channel layout image [B, 4, 64, 64]

        Channel 0: occupancy map
        Channel 1: normalized global feature 1
        Channel 2: normalized global feature 2
        Channel 3: normalized global feature 3

    Output:
        Graph-level scalar prediction [B, 1]
    """

    def __init__(self, in_channels=4):
        super().__init__()

        # -------------------------------------------------
        # CNN feature extractor
        # -------------------------------------------------
        self.cnn = nn.Sequential(

            # [B, 4, 64, 64]
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            # [B, 32, 32, 32]
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            # [B, 64, 16, 16]
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            # [B, 128, 4, 4]
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # -------------------------------------------------
        # Regression head
        # -------------------------------------------------
        self.regressor = nn.Sequential(

            nn.Flatten(),

            # 128 × 4 × 4 = 2048
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.regressor(x)


if __name__ == "__main__":

    model = CNNRegressor(
        in_channels=4
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

    # Test input
    x = torch.randn(1, 4, 64, 64)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)