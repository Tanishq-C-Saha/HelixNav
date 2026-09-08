"""
Observation encoder
"""

import torch
import torch.nn as nn

# from source.helix_nav.helix_nav.tasks.manager_based.navigation.models.encoders.depth_encoder import DepthCNN
# from source.helix_nav.helix_nav.tasks.manager_based.navigation.models.encoders.scalar_encoder import ScalarMLP

from .depth_encoder import DepthCNN
from .scalar_encoder import ScalarMLP


class HelixObservationEncoder(nn.Module):
    """Encodes depth and scalar observations into fused features."""

    def __init__(self) -> None:
        super().__init__()

        # Depth branch
        # (batch, 1, 54, 96) -> (batch, 96)
        self.depth_encoder = DepthCNN(
            height=54,
            width=96,
        )

        # Scalar branch
        # (batch, 41) -> (batch, 96)
        self.scalar_encoder = ScalarMLP()

    def forward(self, depth, scalars) -> torch.Tensor:
        """
        depth:
            (batch, 1, 54, 96)

        scalars:
            (batch, 41)

        output:
            (batch, 192)
        """

        # Encode depth
        depth_features = self.depth_encoder(depth)

        # Encode scalar features
        scalar_features = self.scalar_encoder(scalars)

        # Fuse both feature vectors
        # (batch, 96) + (batch, 96) -> (batch, 192)
        fused_features = torch.cat(
            [depth_features, scalar_features],
            dim=1,
        )

        return fused_features


def main():
    model = HelixObservationEncoder()
    model.to("cuda:0")

    # Dummy inputs
    dummy_depth_input = torch.randn(
        [32, 1, 54, 96],
        device="cuda:0",
    )

    dummy_scalar_input = torch.randn(
        [32, 41],
        device="cuda:0",
    )

    # Forward pass
    output = model(
        depth=dummy_depth_input,
        scalars=dummy_scalar_input,
    )

    print("\n--- Sanity input/output check ---\n")

    print(f"Input Depth Shape   : {dummy_depth_input.shape}")
    print(f"Input Scalar Shape  : {dummy_scalar_input.shape}")
    print(f"Output Shape        : {output.shape}")


if __name__ == "__main__":
    main()
