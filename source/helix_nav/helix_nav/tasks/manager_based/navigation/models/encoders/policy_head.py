"""
Actor and critic heads for HelixNav.
"""

import torch.nn as nn


class ActorHead(nn.Module):
    """
    Maps the GRU's 256-dimensional recurrent
    representation to 3 navigation actions.

    Input:
        (batch, 256)

    Output:
        (batch, 3)

    Actions:
        [v_x, v_y, yaw_rate]
    """

    def __init__(self, input_dim=256, output_dim=3):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),

            nn.Linear(128, 64),
            nn.ELU(),

            nn.Linear(64, output_dim),
            nn.Tanh()  # bounded output [-1,1]
        )

    def forward(self, x):

        assert x.shape[-1] == 256, (
            f"Expected input dim: 256, "
            f"Got: {x.shape[-1]}"
        )

        return self.actor(x)


class CriticHead(nn.Module):
    """
    Maps the GRU's 256-dimensional recurrent
    representation to a scalar value estimate.

    Input:
        (batch, 256)

    Output:
        (batch, 1)
    """

    def __init__(self, input_dim=256):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),

            nn.Linear(128, 64),
            nn.ELU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):

        assert x.shape[-1] == 256, (
            f"Expected input dim: 256, "
            f"Got: {x.shape[-1]}"
        )

        return self.critic(x)