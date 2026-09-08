"""GRU recurrent encoder."""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """Encodes fused 192-dim features into a 256-dim recurrent representation."""

    def __init__(self, input_dim=192, hidden_dim=256, hidden_layers=1):
        super().__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        # Input:  (batch, timestep, 192)
        # Output: (batch, timestep, 256)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.hidden_layers,
            batch_first=True,
        )

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input features (batch, timestep, 192).
            hidden_state: Previous hidden state (layers, batch, 256).

        Returns:
            output: GRU features (batch, timestep, 256).
            hidden_state: Updated hidden state.
        """

        # Check fused feature dimension
        assert x.shape[-1] == self.input_dim, (
            f"Expected input dim: {self.input_dim}, "
            f"got: {x.shape[-1]}"
        )

        # Process current input using previous hidden state
        output, hidden_state = self.gru(
            x,
            hidden_state,
        )

        return output, hidden_state


def main():
    """Run a basic GRU sanity check."""

    model = GRUEncoder().to("cuda:0")

    # 32 environments, 1 timestep, 192 features
    dummy_input = torch.randn(
        [32, 1, 192],
        device="cuda:0",
    )

    output, hidden_state = model(dummy_input)

    print("\n--- GRU Sanity Check ---\n")
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Hidden state : {hidden_state.shape}")


if __name__ == "__main__":
    main()