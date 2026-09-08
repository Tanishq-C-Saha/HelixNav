"""
Depth encoder
"""

import torch
import torch.nn as nn


class DepthCNN(nn.Module):
    """Takes in depth image and produces depth features."""

    def __init__(self, height=54, width=96) -> None:
        super().__init__()

        # CNN encoder
        self.depth_encoder = nn.Sequential(

            # Layer 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2,
            ),

            nn.ELU(),

            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ELU(),

            # Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width)
            conv_out = self.depth_encoder(dummy)
            flat_dim = conv_out.view(1, -1).shape[1]

        # Feature projection
        self.projection = nn.Sequential(

            # Flatten spatial features
            nn.Flatten(),

            # flat_dim -> 256
            nn.Linear(flat_dim, 256),
            nn.ELU(),

            # 256 -> 96
            nn.Linear(256, 96),
            nn.ELU(),
        )

    def forward(self, x):
        """
        input:
            (batch, channel, height=54, width=96)

        output:
            (batch, 96)
        """

        # Check input dimensions
        assert x.shape[1] == 1
        assert x.shape[2] == 54
        assert x.shape[3] == 96

        # Extract spatial features
        x = self.depth_encoder(x)

        # Flatten and project to 96 features
        x = self.projection(x)

        return x


def main():
    model = DepthCNN()
    model.to("cuda:0")

    dummy_input = torch.randn(
        [32, 1, 54, 96],
        device="cuda:0"
    )

    output = model(dummy_input)

    print("\n--- Sanity input/output check ---\n")

    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {output.shape}")

    # Layer-by-layer check
    x = dummy_input

    print("\n--- Sanity Layer by Layer check ---\n")
    print(f"Input Shape : {tuple(x.shape)}")

    for i, layer in enumerate(model.depth_encoder):

        x = layer(x)

        print(
            f"Layer {i} "
            f"{layer.__class__.__name__:<10}"
            f" -> {tuple(x.shape)}"
        )


if __name__ == "__main__":
    main()
