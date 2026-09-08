"""
Scalar encoder
"""

import torch
import torch.nn as nn


class ScalarMLP(nn.Module):
    """Takes scalar observations and produces scalar features."""

    def __init__(self) -> None:
        super().__init__()

        # MLP encoder
        self.scalar_encoder = nn.Sequential(

            # Layer 1
            # 41 -> 128
            nn.Linear(41, 128),
            nn.ELU(),

            # Layer 2
            # 128 -> 128
            nn.Linear(128, 128),
            nn.ELU(),

            # Layer 3
            # 128 -> 96
            nn.Linear(128, 96),
            nn.ELU(),
        )

    def forward(self, x):
        """
        input:
            (batch, 41)

        output:
            (batch, 96)
        """

        # Check input dimension
        assert x.shape[-1] == 41

        # Encode scalar features
        return self.scalar_encoder(x)


def main():
    model = ScalarMLP()
    model.to("cuda:0")

    dummy_input = torch.randn(
        [32, 41],
        device="cuda:0",
    )

    output = model(dummy_input)

    print("\n--- Sanity input/output check ---\n")

    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {output.shape}")

    # Layer-by-layer check
    x = dummy_input

    print("\n--- Sanity Layer by Layer check ---\n")
    print(f"Input Shape : {tuple(x.shape)}")

    for i, layer in enumerate(model.scalar_encoder):

        x = layer(x)

        print(
            f"Layer {i} "
            f"{layer.__class__.__name__:<10}"
            f" -> {tuple(x.shape)}"
        )


if __name__ == "__main__":
    main()