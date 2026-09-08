"""HelixNav Rl policy"""

import torch
import torch.nn as nn

from encoders import (
    HelixObservationEncoder,
    GRUEncoder,
    ActorHead,
    CriticHead
)


class HelixNavPolicy(nn.Module):
    """HelixNav Policy"""

    def __init__(self):
        super().__init__()

        self.obs_encoder = HelixObservationEncoder()
        self.gru = GRUEncoder(input_dim=192, hidden_dim=256)
        self.actor = ActorHead(input_dim=256)
        self.critic = CriticHead(input_dim=256)

    def forward(self, depth, scalars, hidden_state=None):
        """
        
        """
        # encode observations
        fused = self.obs_encoder(depth, scalars)          # (N, 192)

        # add time dimension for GRU
        fused = fused.unsqueeze(1)                     # (N, 1, 192)

        # recurrent processing
        gru_out, new_hidden = self.gru(fused, hidden_state)
        gru_out = gru_out.squeeze(1)                   # (N, 256)

        # heads
        actions = self.actor(gru_out)                  # (N, 3)
        value = self.critic(gru_out)                   # (N, 1)

        return actions, value, new_hidden


    def init_hidden(self, num_envs, device):
        """Full zero init, call once at training start."""
        return torch.zeros(
            self.gru.hidden_layers, num_envs, self.gru.hidden_dim,
            device=device
        )

    def reset_hidden(self, hidden_state, env_ids):
        """Zero only the envs that just reset, leave others untouched."""
        hidden_state[:, env_ids, :] = 0.0
        return hidden_state


def main():
    model = HelixNavPolicy()
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
    actor_output, critic_output, new_hidden = model(
        depth=dummy_depth_input,
        scalars=dummy_scalar_input,
    )

    print("\n--- Sanity input/output check ---\n")

    print(f"Input Depth Shape       : {tuple(dummy_depth_input.shape)}")
    print(f"Input Scalar Shape      : {tuple(dummy_scalar_input.shape)}")
    print(f"Actor Output Shape      : {tuple(actor_output.shape)}")
    print(f"Critic Output Shape     : {tuple(critic_output.shape)}")
    print(f"New Hidden Output Shape : {tuple(new_hidden.shape)}")


if __name__ == "__main__":
    main()
