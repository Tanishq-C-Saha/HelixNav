"""skrl policy wrapper for HelixNav CP7.

THIN ADAPTER — all neural network logic lives in the encoders/ and heads/ sub-modules.
This file only translates between skrl's recurrent PPO interface and our PyTorch modules.

Architecture:
    Flat obs → slice depth + scalars → DepthCNN + ScalarMLP → fuse 192 → GRU 256 → Actor/Critic heads

Key skrl interface points:
    - compute(inputs, role) → (output, {"log_std": ..., "rnn": [hidden_states]})
    - get_specification() → {"rnn": {"sequence_length": L, "sizes": [(layers, N, hidden)]}}
    - Training: inputs["observations"] reshaped to (batch, seq_len, features)
    - Rollout: inputs["observations"] is (N, features) — single timestep
    - inputs["rnn"][0] = GRU hidden state
    - inputs["terminated"] = episode boundary mask (training only)
"""

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from .encoders.depth_encoder import DepthCNN
from .encoders.scalar_encoder import ScalarMLP
from .encoders.gru_encoder import GRUEncoder
from .encoders.policy_head import ActorHead, CriticHead


# ── Observation layout (must match ObservationsCfg.policy field order — see
# manager_configs/observations.py; concatenate_terms=True there) ──
DEPTH_H, DEPTH_W = 54, 96
DEPTH_FLAT = DEPTH_H * DEPTH_W  # 5184 (already flattened by a modifier in the obs term)

SCALAR_SIZES = [24, 8, 3, 3, 3]  # lookahead_vectors, snap_flags, base_velocity, prev_actions, relative_goal
SCALAR_DIM = sum(SCALAR_SIZES)  # 41
TOTAL_FLAT = DEPTH_FLAT + SCALAR_DIM  # 5225

FUSED_DIM = 192  # 96 (depth) + 96 (scalars)
GRU_HIDDEN = 256
NUM_ACTIONS = 3


class SharedEncoder(nn.Module):
    """Shared encoder + GRU backbone used by both actor and critic.

    Keeps one copy of the heavy computation (CNN + MLP + GRU) to save VRAM.
    """

    def __init__(self):
        super().__init__()
        self.depth_cnn = DepthCNN(height=DEPTH_H, width=DEPTH_W)
        self.scalar_mlp = ScalarMLP()
        self.gru = GRUEncoder(input_dim=FUSED_DIM, hidden_dim=GRU_HIDDEN)

    def forward(self, depth, scalars, hidden_state, training, sequence_length, terminated):
        """Run encoder + GRU with proper sequence handling.

        Args:
            depth:     (B, 1, 54, 96) or (B*L, 1, 54, 96) during training
            scalars:   (B, 41) or (B*L, 41) during training
            hidden_state: (1, N, 256) GRU hidden
            training:  bool — are we in PPO training or rollout collection
            sequence_length: int — L for training sequences
            terminated: (B*L,) or None — episode boundary mask

        Returns:
            gru_features: (B*L, 256) or (N, 256)
            new_hidden:   (1, N, 256)
        """

        # ── encode observations ──
        depth_features = self.depth_cnn(depth)     # (B*L, 96) or (N, 96)
        scalar_features = self.scalar_mlp(scalars)  # (B*L, 96) or (N, 96)
        fused = torch.cat([depth_features, scalar_features], dim=-1)  # (B*L, 192) or (N, 192)

        # ── GRU with sequence handling ──
        #
        # cuDNN disabled around this GRU specifically: skrl's PPO_RNN.update() calls
        # self.policy.act(...) and then self.value.act(...) within the same mini-batch
        # iteration, before one combined .backward(). Since actor/critic share this one
        # GRU module (SharedEncoder), that means the SAME nn.GRU instance gets a forward
        # pass twice before backward. cuDNN's fused RNN kernel keeps its backward
        # reserve-space/workspace tied to the module (not the call), and the second
        # forward call overwrites what the first call's backward needs — raising
        # "RuntimeError: cudnn RNN backward can only be called in training mode" even
        # though training mode is set correctly throughout. This is a documented cuDNN
        # RNN limitation; ordinary (non-RNN) layers like Conv2d/Linear don't have it —
        # calling the same conv/linear module twice before one backward is normal and
        # well-supported (e.g. siamese networks). PyTorch's native (non-cuDNN) GRU
        # implementation has no such per-module reserve-space coupling, so disabling
        # cuDNN just for this call sidesteps the corruption without needing two separate
        # GRU instances (which would mean manually keeping weights tied across them).
        # The GRU is a small fraction of this model's compute (the depth CNN over
        # 54x96 images dominates), so the performance cost here is minor.
        with torch.backends.cudnn.flags(enabled=False):
            if training:
                # reshape fused features into sequences
                rnn_input = fused.view(-1, sequence_length, FUSED_DIM)  # (batch, L, 192)

                # reshape hidden state: (layers, batch*L, hidden) → take initial step
                hidden_state = hidden_state.view(
                    self.gru.hidden_layers, -1, sequence_length, GRU_HIDDEN
                )
                hidden_state = hidden_state[:, :, 0, :].contiguous()  # (layers, batch, hidden)

                # handle episode resets mid-sequence
                if terminated is not None and torch.any(terminated):
                    rnn_outputs = []
                    terminated_seq = terminated.view(-1, sequence_length)

                    # find indices where resets happen within the sequence
                    indexes = (
                        [0]
                        + (terminated_seq[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                        + [sequence_length]
                    )

                    for i in range(len(indexes) - 1):
                        i0, i1 = indexes[i], indexes[i + 1]
                        rnn_output, hidden_state = self.gru.gru(
                            rnn_input[:, i0:i1, :], hidden_state
                        )
                        # zero hidden state for envs that terminated at the boundary
                        hidden_state[:, terminated_seq[:, i1 - 1], :] = 0
                        rnn_outputs.append(rnn_output)

                    rnn_output = torch.cat(rnn_outputs, dim=1)
                else:
                    rnn_output, hidden_state = self.gru.gru(rnn_input, hidden_state)

            else:
                # rollout: single timestep
                rnn_input = fused.view(-1, 1, FUSED_DIM)  # (N, 1, 192)
                rnn_output, hidden_state = self.gru.gru(rnn_input, hidden_state)

        # flatten: (batch, L, 256) → (batch*L, 256)
        gru_features = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return gru_features, hidden_state


def _split_observations(observations):
    """Split a flat (B, 5225) observation tensor into depth and scalar tensors.

    Slice order matches ObservationsCfg.policy field order (concatenate_terms=True):
    depth_images | lookahead_vectors | snap_flags | base_velocity | prev_actions | relative_goal.
    Works identically for rollout (B=N) and training (B=N*L) batches — skrl flattens
    the sequence dimension into the batch dimension before calling compute().

    Returns:
        depth:   (B, 1, H, W)
        scalars: (B, 41)
    """
    if observations.shape[-1] != TOTAL_FLAT:
        raise ValueError(
            f"Expected flat observations of size {TOTAL_FLAT} (depth {DEPTH_FLAT} + scalars {SCALAR_DIM}), "
            f"got {observations.shape[-1]}. Check ObservationsCfg.policy field order/sizes match this module."
        )

    depth = observations[:, :DEPTH_FLAT].view(-1, 1, DEPTH_H, DEPTH_W)
    scalars = observations[:, DEPTH_FLAT:]

    return depth, scalars


# ──────────────────────────────────────────────
#  Actor (stochastic policy)
# ──────────────────────────────────────────────

class HelixNavActorRNN(GaussianMixin, Model):
    """skrl actor: GaussianMixin for stochastic PPO actions."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        shared_encoder: SharedEncoder,
        num_envs=1,
        sequence_length=30,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-5.0,
        max_log_std=2.0,
        initial_log_std=-1.0,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.num_envs = num_envs
        self.sequence_length = sequence_length

        # shared encoder + GRU (same instance as critic)
        self.encoder = shared_encoder

        # actor head: GRU output → mean actions
        self.actor_head = ActorHead(input_dim=GRU_HIDDEN, output_dim=NUM_ACTIONS)

        # learnable log_std
        self.log_std_parameter = nn.Parameter(
            torch.full((NUM_ACTIONS,), initial_log_std, device=device)
        )

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (1, self.num_envs, GRU_HIDDEN),  # GRU: 1 hidden state
                ],
            }
        }

    def compute(self, inputs, role=""):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # split flat obs into depth + scalars
        depth, scalars = _split_observations(observations)

        # shared encoder + GRU
        gru_features, new_hidden = self.encoder(
            depth=depth,
            scalars=scalars,
            hidden_state=hidden_states,
            training=self.training,
            sequence_length=self.sequence_length,
            terminated=terminated,
        )

        # actor head → mean actions
        mean_actions = self.actor_head(gru_features)

        return mean_actions, {"log_std": self.log_std_parameter, "rnn": [new_hidden]}


# ──────────────────────────────────────────────
#  Critic (value function)
# ──────────────────────────────────────────────

class HelixNavCriticRNN(DeterministicMixin, Model):
    """skrl critic: DeterministicMixin for value estimation."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        shared_encoder: SharedEncoder,
        num_envs=1,
        sequence_length=30,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        self.num_envs = num_envs
        self.sequence_length = sequence_length

        # shared encoder + GRU (same instance as actor)
        self.encoder = shared_encoder

        # critic head: GRU output → scalar value
        self.critic_head = CriticHead(input_dim=GRU_HIDDEN)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (1, self.num_envs, GRU_HIDDEN),
                ],
            }
        }

    def compute(self, inputs, role=""):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        depth, scalars = _split_observations(observations)

        gru_features, new_hidden = self.encoder(
            depth=depth,
            scalars=scalars,
            hidden_state=hidden_states,
            training=self.training,
            sequence_length=self.sequence_length,
            terminated=terminated,
        )

        value = self.critic_head(gru_features)

        return value, {"rnn": [new_hidden]}


# ──────────────────────────────────────────────
#  Factory function
# ──────────────────────────────────────────────

def build_models(env, sequence_length=30):
    """Create actor + critic with shared encoder and GRU.

    Args:
        env: Isaac Lab ManagerBasedRLEnv (wrapped for skrl)
        sequence_length: RNN training chunk length (30 ≈ 3 sec at 10Hz)

    Returns:
        dict: {"policy": actor, "value": critic}
    """
    device = env.device
    num_envs = env.num_envs

    # shared backbone — same weights for actor and critic
    shared = SharedEncoder().to(device)

    actor = HelixNavActorRNN(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
        shared_encoder=shared,
        num_envs=num_envs,
        sequence_length=sequence_length,
    )

    critic = HelixNavCriticRNN(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
        shared_encoder=shared,
        num_envs=num_envs,
        sequence_length=sequence_length,
    )

    return {"policy": actor, "value": critic}


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import gymnasium as gym

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = 4  # num envs

    # mock observation space matching our env (flat, concatenate_terms=True)
    obs_space = gym.spaces.Box(-50.0, 50.0, shape=(TOTAL_FLAT,))
    act_space = gym.spaces.Box(-1, 1, shape=(NUM_ACTIONS,))
    state_space = obs_space  # symmetric actor-critic

    shared = SharedEncoder().to(device)

    actor = HelixNavActorRNN(
        observation_space=obs_space,
        state_space=state_space,
        action_space=act_space,
        device=device,
        shared_encoder=shared,
        num_envs=N,
        sequence_length=10,
    )

    critic = HelixNavCriticRNN(
        observation_space=obs_space,
        state_space=state_space,
        action_space=act_space,
        device=device,
        shared_encoder=shared,
        num_envs=N,
        sequence_length=10,
    )

    print(f"Actor spec: {actor.get_specification()}")
    print(f"Critic spec: {critic.get_specification()}")

    # ── rollout mode test ──
    actor.eval()
    critic.eval()

    dummy_obs = torch.randn(N, TOTAL_FLAT, device=device)
    hidden = torch.zeros(1, N, GRU_HIDDEN, device=device)

    inputs = {"observations": dummy_obs, "rnn": [hidden]}

    actions, actor_out = actor.compute(inputs)
    values, critic_out = critic.compute(inputs)

    print(f"\n--- Rollout mode ---")
    print(f"Actions shape:      {actions.shape}")  # (N, 3)
    print(f"Values shape:       {values.shape}")   # (N, 1)
    print(f"Hidden shape:       {actor_out['rnn'][0].shape}")  # (1, N, 256)
    print(f"log_std:            {actor_out['log_std']}")

    # verify GRU is actually recurrent
    _, out1 = actor.compute(inputs)
    hidden_modified = hidden.clone()
    hidden_modified[:, 0, :] = 5.0  # change env 0's hidden state
    inputs2 = {"observations": dummy_obs, "rnn": [hidden_modified]}
    _, out2 = actor.compute(inputs2)

    diff = (out1["rnn"][0] - out2["rnn"][0]).abs().sum().item()
    print(f"\nHidden state diff with modified input: {diff:.4f}")
    assert diff > 0, "GRU is not actually using hidden state!"

    print("\n✅ skrl wrapper verified")