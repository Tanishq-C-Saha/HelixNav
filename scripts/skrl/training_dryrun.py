# Copyright (c) 2026, HelixNav Project
# SPDX-License-Identifier: BSD-3-Clause

"""
HelixNav Training Dry-Run
==========================

Runs a handful of *actual* PPO training iterations (rollout collection +
gradient update), not just env steps, to verify the full agent pipeline is
wired correctly before committing to a long run. Unlike preflight_check.py
(which only exercises the environment), this instantiates the PPO_RNN agent
directly — bypassing skrl's Runner/Trainer — so internals normally hidden
from view can be inspected:
  - policy/value/entropy loss finiteness
  - gradient flow (dead/missing gradients, gradient norm stats)
  - action distribution collapse
  - GRU hidden state shape, movement, and reset-on-episode-boundary behaviour
  - value function sanity
  - shared encoder wiring between actor and critic
  - actual parameter updates from the optimizer step

Usage:
    python scripts/skrl/training_dryrun.py --task HelixNav-CP7-v0 --num_envs 64
    python scripts/skrl/training_dryrun.py --task HelixNav-CP7-v0 --num_envs 64 --iterations 5

Exit codes:
    0 = all checks passed
    1 = at least one FAIL
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import traceback
from dataclasses import dataclass, field

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HelixNav training dry-run (smoke test).")
parser.add_argument("--task", type=str, required=True, help="Registered gym task ID.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--iterations", type=int, default=3, help="Number of PPO training iterations to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below runs after sim app is up."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO_RNN, PPO_CFG
from skrl.memories.torch import RandomMemory

import helix_nav.tasks  # noqa: F401 — triggers gym.register

from helix_nav.tasks.manager_based.navigation.models.skrl_policy import build_models, GRU_HIDDEN


# ════════════════════════════════════════════════
#  Result tracking (same pattern as preflight_check.py)
# ════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    warn: bool = False


@dataclass
class Report:
    results: list = field(default_factory=list)

    def add(self, name, passed, detail="", warn=False):
        self.results.append(CheckResult(name, passed, detail, warn))

    @property
    def all_passed(self):
        return all(r.passed for r in self.results)

    def print_summary(self):
        print("\n" + "=" * 72)
        print("  TRAINING DRY-RUN SUMMARY")
        print("=" * 72)

        fails = [r for r in self.results if not r.passed]
        warns = [r for r in self.results if r.passed and r.warn]
        passes = [r for r in self.results if r.passed and not r.warn]

        for r in passes:
            print(f"  ✅ PASS  {r.name}")
            if r.detail:
                for line in r.detail.split("\n"):
                    print(f"           {line}")

        for r in warns:
            print(f"  ⚠️  WARN  {r.name}")
            if r.detail:
                for line in r.detail.split("\n"):
                    print(f"           {line}")

        for r in fails:
            print(f"  ❌ FAIL  {r.name}")
            if r.detail:
                for line in r.detail.split("\n"):
                    print(f"           {line}")

        print("=" * 72)
        n = len(self.results)
        print(f"  {len(passes)} passed, {len(warns)} warnings, {len(fails)} failed out of {n} checks")

        if fails:
            print("  ❌ DO NOT TRAIN — fix failures first")
        elif warns:
            print("  ⚠️  Trainable but review warnings")
        else:
            print("  ✅ All clear — safe to train")
        print("=" * 72 + "\n")


# ════════════════════════════════════════════════
#  Static (one-time) checks
# ════════════════════════════════════════════════

def check_shared_encoder(actor, critic, report):
    """Confirm actor and critic actually share one encoder instance, not copies."""
    same_object = actor.encoder is critic.encoder
    if not same_object:
        report.add(
            "shared_encoder_identity", False,
            "actor.encoder is not critic.encoder — VRAM will be doubled and the two "
            "networks will learn divergent representations."
        )
        return

    actor_encoder_params = {id(p) for p in actor.encoder.parameters()}
    critic_encoder_params = {id(p) for p in critic.encoder.parameters()}
    actor_all_params = {id(p) for p in actor.parameters()}
    critic_all_params = {id(p) for p in critic.parameters()}

    encoder_in_actor = actor_encoder_params <= actor_all_params
    encoder_in_critic = critic_encoder_params <= critic_all_params
    shared_params_match = actor_encoder_params == critic_encoder_params

    if encoder_in_actor and encoder_in_critic and shared_params_match:
        report.add(
            "shared_encoder_identity", True,
            f"actor.encoder is critic.encoder (same object). "
            f"{len(actor_encoder_params)} shared encoder params visible in both param lists."
        )
    else:
        report.add(
            "shared_encoder_identity", False,
            f"Encoder is the same object but its params are not fully visible via "
            f".parameters(): in_actor={encoder_in_actor}, in_critic={encoder_in_critic}, "
            f"match={shared_params_match}."
        )


def check_optimizer_param_duplication(agent, report):
    """Detect the same parameter tensor listed twice in the optimizer.

    PPO_RNN builds its optimizer as
    ``Adam(itertools.chain(policy.parameters(), value.parameters()), ...)``
    whenever ``policy is not value``. If policy and value share a submodule
    (our shared encoder), that submodule's parameters end up in the flattened
    params list twice. PyTorch's duplicate-parameter guard only checks across
    separate ``param_groups``, not within a single list passed at construction,
    so this goes undetected — and ``optimizer.step()`` then applies the Adam
    update twice per step to every shared-encoder parameter, silently doubling
    its effective learning rate and corrupting its momentum state.
    """
    seen = set()
    dupes = 0
    total = 0
    for group in agent.optimizer.param_groups:
        for p in group["params"]:
            total += 1
            pid = id(p)
            if pid in seen:
                dupes += 1
            seen.add(pid)

    if dupes > 0:
        report.add(
            "optimizer_param_duplication", False,
            f"{dupes}/{total} optimizer parameter entries are duplicate tensors — almost "
            f"certainly the shared encoder appearing in both policy.parameters() and "
            f"value.parameters(). Each optimizer.step() call will update these params twice, "
            f"doubling their effective learning rate."
        )
    else:
        report.add("optimizer_param_duplication", True, f"No duplicate parameter references in optimizer ({total} params)")


def build_deduplicated_optimizer(actor, critic, learning_rate):
    """Build an Adam optimizer that lists the shared encoder's params exactly once.

    Replaces PPO_RNN's default ``Adam(chain(policy.parameters(), value.parameters()))``,
    which double-lists every shared-encoder parameter (see
    ``check_optimizer_param_duplication``) and would silently double their effective
    learning rate and corrupt Adam's momentum state for those params.

    PPO_CFG (this skrl version) has no hook to inject a pre-built optimizer, so the
    caller is expected to overwrite ``agent.optimizer`` (and
    ``agent.checkpoint_modules["optimizer"]``) with the result right after
    constructing the agent, before any rollout/update runs.
    """
    encoder_param_ids = {id(p) for p in actor.encoder.parameters()}
    actor_only_params = [p for p in actor.parameters() if id(p) not in encoder_param_ids]
    critic_only_params = [p for p in critic.parameters() if id(p) not in encoder_param_ids]

    return torch.optim.Adam(
        [
            {"params": list(actor.encoder.parameters())},  # shared encoder — listed once
            {"params": actor_only_params},  # actor head + log_std
            {"params": critic_only_params},  # critic head
        ],
        lr=learning_rate,
    )


def neutralize_shared_encoder_training_toggle(critic):
    """Work around a cuDNN RNN + shared-encoder incompatibility in PPO_RNN.update().

    Before its mini-batch loop, PPO_RNN.update() computes a bootstrap value estimate
    under torch.no_grad() by toggling ONLY the critic to eval mode and back:
        self.value.enable_training_mode(False)
        last_values, _ = self.value.act(inputs, role="value")
        self.value.enable_training_mode(True)
    Because our critic's GRU is the SAME object as the actor's (SharedEncoder), this
    transiently drops the shared cuDNN GRU into eval mode mid-update. PyTorch's cuDNN
    RNN backend corrupts the reserve-space it needs for backward when a module runs in
    eval mode between training-mode forward passes that get backpropagated together —
    a documented cuDNN RNN gotcha, reported repeatedly for actor-critic setups with
    shared encoders. It surfaces as:
        RuntimeError: cudnn RNN backward can only be called in training mode
    at the .backward() call in update(), even though training mode is correctly
    restored to True before the mini-batch loop runs — the corruption already
    happened during the earlier eval-mode forward on the same cuDNN module.

    There are two independent "training" signals in play, which must NOT be conflated:
      1. critic.training (this wrapper's own nn.Module flag) — HelixNavCriticRNN.compute()
         passes it explicitly as SharedEncoder.forward()'s `training` argument, which
         selects the Python branch (single-timestep reshape vs. sequence-batch reshape).
         This MUST keep toggling normally (False during rollout, True during update()),
         or record_transition()'s single-timestep value() call takes the wrong branch
         and crashes with a shape mismatch.
      2. critic.encoder.gru.gru.training (the underlying cuDNN GRU module's own flag,
         shared with the actor since it's the same object) — this is what must never
         flip to eval mode between two training-mode forward passes that share a later
         backward(). skrl's toggle assumes policy and value never share a module, so
         this asymmetric toggle was always safe for skrl's intended (non-shared) use case.

    So: let critic.training keep toggling correctly, but skip recursing into the shared
    `encoder` child specifically — its train/eval state is driven exclusively by the
    actor's own (unpatched) calls, which happen in lockstep with the critic's at every
    other call site (Agent.enable_models_training_mode toggles both together). Toggling
    encoder.training=False would have no effect on the *values* SharedEncoder.forward()
    computes (no dropout/batchnorm in it) — cuDNN's training flag only affects whether it
    retains a reserve-space buffer for backward, a pure perf/memory concern, not accuracy.
    """
    def _enable_training_mode(enabled=True):
        critic.training = enabled
        for name, module in critic.named_children():
            if name == "encoder":
                continue
            module.train(enabled)

    critic.enable_training_mode = _enable_training_mode


def check_rnn_spec(actor, critic, num_envs, report):
    """Verify get_specification() reports the expected RNN shape."""
    issues = []
    for name, model in [("actor", actor), ("critic", critic)]:
        spec = model.get_specification()
        sizes = spec.get("rnn", {}).get("sizes", [])
        if not sizes:
            issues.append(f"{name}: get_specification() returned no rnn sizes")
            continue
        expected = (1, num_envs, GRU_HIDDEN)
        if tuple(sizes[0]) != expected:
            issues.append(f"{name}: rnn size {tuple(sizes[0])} != expected {expected}")

    if issues:
        report.add("rnn_specification", False, "\n".join(issues))
    else:
        report.add("rnn_specification", True, f"Both actor and critic report rnn size (1, {num_envs}, {GRU_HIDDEN})")


# ════════════════════════════════════════════════
#  Per-iteration checks
# ════════════════════════════════════════════════

def snapshot_params(model):
    """Sum of absolute parameter values, used as a cheap before/after checksum."""
    with torch.no_grad():
        return {name: p.detach().clone() for name, p in model.named_parameters()}


def check_gradient_flow(model, name, report):
    """After a backward pass, check for missing/dead gradients across all params."""
    total = 0
    no_grad = 0
    zero_grad = 0
    norms = []

    for pname, p in model.named_parameters():
        total += 1
        if p.grad is None:
            no_grad += 1
            continue
        norm = p.grad.norm().item()
        norms.append(norm)
        if norm == 0.0:
            zero_grad += 1

    if total == 0:
        report.add(f"gradient_flow_{name}", False, "Model has no parameters")
        return

    dead_frac = (no_grad + zero_grad) / total
    detail = (
        f"{total} params: {no_grad} with grad=None, {zero_grad} with zero-norm grad "
        f"({dead_frac * 100:.1f}% dead)."
    )
    if norms:
        detail += f"\n  grad norm: min={min(norms):.6f}  max={max(norms):.6f}  mean={sum(norms) / len(norms):.6f}"

    if dead_frac > 0.10:
        report.add(f"gradient_flow_{name}", False, detail)
    else:
        report.add(f"gradient_flow_{name}", True, detail, warn=(dead_frac > 0.0))


def check_param_update(before, after, name, report):
    """Confirm the optimizer actually changed the parameters."""
    unchanged = []
    for pname, before_val in before.items():
        after_val = after[pname]
        if torch.equal(before_val, after_val):
            unchanged.append(pname)

    total = len(before)
    if unchanged:
        pct = 100.0 * len(unchanged) / total
        detail = f"{len(unchanged)}/{total} ({pct:.1f}%) params unchanged after optimizer step"
        if len(unchanged) == total:
            report.add(f"param_update_{name}", False, detail + " — optimizer did not update anything")
        else:
            detail += "\n  unchanged: " + ", ".join(unchanged[:10]) + (" ..." if len(unchanged) > 10 else "")
            report.add(f"param_update_{name}", True, detail, warn=True)
    else:
        report.add(f"param_update_{name}", True, f"All {total} params changed after optimizer step")


def check_losses(tracking_data, iteration, report):
    """Loss sanity: policy/value/entropy losses must be finite."""
    keys = {
        "Loss / Policy loss": "policy_loss",
        "Loss / Value loss": "value_loss",
        "Loss / Entropy loss": "entropy_loss",
    }

    issues = []
    values = {}
    for tb_key, short in keys.items():
        series = tracking_data.get(tb_key)
        if not series:
            if short == "entropy_loss":
                continue  # entropy loss is optional (entropy_loss_scale may be 0)
            issues.append(f"{short}: no data tracked")
            continue
        val = series[-1]
        values[short] = val
        if val != val or val in (float("inf"), float("-inf")):  # NaN check via val != val
            issues.append(f"{short}: non-finite ({val})")

    detail = ", ".join(f"{k}={v:.6f}" for k, v in values.items())
    if issues:
        report.add(f"iter{iteration}_loss_sanity", False, "\n".join(issues) + f"\n{detail}")
    else:
        report.add(f"iter{iteration}_loss_sanity", True, detail)


def check_action_distribution(actions_history, iteration, report):
    """actions_history: list of (num_envs, 3) tensors collected across one rollout."""
    stacked = torch.stack(actions_history)  # (steps, num_envs, 3)

    per_env_std = stacked.std(dim=0)  # variability across time, per env/action-dim
    collapsed = (per_env_std < 0.01).float().mean().item()

    # check timesteps aren't just repeating the same action
    diffs = (stacked[1:] - stacked[:-1]).abs().mean().item()

    issues = []
    if collapsed > 0.5:
        issues.append(f"{collapsed * 100:.1f}% of (env, action-dim) pairs have std < 0.01 across the rollout")
    if diffs < 1e-5:
        issues.append(f"Mean |action[t] - action[t-1]| = {diffs:.2e} — actions appear frozen across timesteps")

    detail = (
        f"action std: min={per_env_std.min().item():.4f} max={per_env_std.max().item():.4f} "
        f"mean={per_env_std.mean().item():.4f}\n"
        f"mean |Δaction| between steps: {diffs:.4f}"
    )

    if issues:
        report.add(f"iter{iteration}_action_distribution", False, "\n".join(issues) + "\n" + detail)
    else:
        report.add(f"iter{iteration}_action_distribution", True, detail)


def check_hidden_state(hidden_history, num_envs, iteration, report):
    """hidden_history: list of (1, num_envs, GRU_HIDDEN) tensors collected across one rollout."""
    issues = []

    for h in hidden_history:
        if tuple(h.shape) != (1, num_envs, GRU_HIDDEN):
            issues.append(f"Unexpected hidden state shape {tuple(h.shape)}, expected (1, {num_envs}, {GRU_HIDDEN})")
            break

    if len(hidden_history) >= 2:
        deltas = [(hidden_history[i] - hidden_history[i - 1]).abs().mean().item() for i in range(1, len(hidden_history))]
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta < 1e-6:
            issues.append(f"Hidden state barely changes between steps (mean |Δh|={mean_delta:.2e}) — GRU may be frozen")
    else:
        mean_delta = float("nan")

    detail = f"{len(hidden_history)} steps observed. mean |Δh| between steps: {mean_delta:.6f}"

    if issues:
        report.add(f"iter{iteration}_hidden_state", False, "\n".join(issues) + "\n" + detail)
    else:
        report.add(f"iter{iteration}_hidden_state", True, detail)


def check_hidden_reset(hidden_after_reset, reset_env_ids, iteration, report):
    """Verify hidden state zeroes out for envs whose episode just ended."""
    if reset_env_ids.numel() == 0:
        report.add(f"iter{iteration}_hidden_reset", True, "No episodes ended this iteration — nothing to check", warn=True)
        return

    reset_hidden = hidden_after_reset[:, reset_env_ids, :]
    max_abs = reset_hidden.abs().max().item()

    if max_abs > 1e-6:
        report.add(
            f"iter{iteration}_hidden_reset", False,
            f"{reset_env_ids.numel()} env(s) terminated/truncated but their hidden state was not zeroed "
            f"(max |h|={max_abs:.4f})"
        )
    else:
        report.add(
            f"iter{iteration}_hidden_reset", True,
            f"Hidden state correctly zeroed for {reset_env_ids.numel()} env(s) that ended an episode"
        )


def check_values(stacked, iteration, report):
    """stacked: (steps, num_envs, 1) tensor of value estimates recorded into agent memory."""
    issues = []
    if torch.isnan(stacked).any():
        issues.append(f"NaN in values: {torch.isnan(stacked).sum().item()} entries")
    if torch.isinf(stacked).any():
        issues.append(f"Inf in values: {torch.isinf(stacked).sum().item()} entries")

    std = stacked.std().item()
    if std < 1e-6:
        issues.append(f"Value estimates have near-zero variance (std={std:.2e}) — critic may not be learning anything")

    detail = f"value stats: min={stacked.min().item():.4f} max={stacked.max().item():.4f} std={std:.4f}"

    if issues:
        report.add(f"iter{iteration}_value_function", False, "\n".join(issues) + "\n" + detail)
    else:
        report.add(f"iter{iteration}_value_function", True, detail)


# ════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════

def main():
    report = Report()

    print("\n" + "=" * 72)
    print(f"  HelixNav Training Dry-Run: {args_cli.task}")
    print(f"  num_envs={args_cli.num_envs}  seed={args_cli.seed}  iterations={args_cli.iterations}")
    print("=" * 72 + "\n")

    rollouts = 30  # matches sequence_length so exactly one PPO update per iteration

    # ── create + wrap environment ──
    print("[setup] Creating environment...")
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
        env_cfg.seed = args_cli.seed
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")
    except Exception as e:
        report.add("env_creation", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
        report.print_summary()
        return 1
    report.add("env_creation", True, f"Created {args_cli.num_envs} envs on {env.device}")

    # ── build models ──
    print("[setup] Building actor/critic models...")
    try:
        models = build_models(env, sequence_length=rollouts)
        actor, critic = models["policy"], models["value"]
    except Exception as e:
        report.add("model_build", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
        report.print_summary()
        env.close()
        return 1
    report.add("model_build", True, "Actor and critic constructed with shared encoder")

    check_shared_encoder(actor, critic, report)
    check_rnn_spec(actor, critic, env.num_envs, report)

    # ── build PPO_RNN agent (manual instantiation, no Runner/Trainer) ──
    print("[setup] Building PPO_RNN agent...")
    learning_rate = 3e-4
    try:
        memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

        agent_cfg = PPO_CFG(
            rollouts=rollouts,
            learning_epochs=2,
            mini_batches=2,
            learning_rate=learning_rate,
            entropy_loss_scale=0.01,  # nonzero so check_losses also exercises entropy-loss tracking
            time_limit_bootstrap=True,  # bootstrap value at truncation, not just termination
            experiment={"write_interval": 0, "checkpoint_interval": 0},
        )

        agent = PPO_RNN(
            models=models,
            memory=memory,
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
            cfg=agent_cfg,
        )

        # PPO_RNN's default optimizer (Adam(chain(policy.parameters(), value.parameters())))
        # double-lists every shared-encoder parameter since actor/critic share one encoder.
        # PPO_CFG has no hook to inject a pre-built optimizer, so replace it post-construction
        # — safe here since our cfg sets no learning_rate_scheduler (agent.scheduler is None,
        # so nothing else still references the old optimizer object).
        agent.optimizer = build_deduplicated_optimizer(actor, critic, learning_rate)
        agent.checkpoint_modules["optimizer"] = agent.optimizer
        neutralize_shared_encoder_training_toggle(critic)

        agent.init(trainer_cfg=None)
        # Only flip the agent's own `training` flag — this is what post_interaction()
        # gates its `update()` call on. Do NOT pass apply_to_models=True: that would
        # force actor/critic into nn.Module training mode permanently, so every act()
        # call (including plain rollout steps) would hit SharedEncoder.forward's
        # training-time sequence reshape instead of the single-timestep rollout path.
        # PPO_RNN.post_interaction() already toggles model training mode itself,
        # correctly, only around the actual update() call.
        agent.enable_training_mode(True)
    except Exception as e:
        report.add("agent_build", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
        report.print_summary()
        env.close()
        return 1
    report.add("agent_build", True, f"PPO_RNN agent built (rollouts={rollouts}, mini_batches=2, lr={learning_rate})")
    check_optimizer_param_duplication(agent, report)

    # ── manual rollout + update loop ──
    observations, _ = env.reset()
    timesteps = rollouts * args_cli.iterations
    timestep = 0

    try:
        for iteration in range(args_cli.iterations):
            print(f"[{iteration + 1}/{args_cli.iterations}] Collecting rollout + training...")

            actions_history = []
            hidden_history = []
            last_reset_env_ids = torch.empty(0, dtype=torch.long, device=env.device)
            last_hidden_after_reset = None

            actor_before = snapshot_params(actor)
            critic_before = snapshot_params(critic)

            for _ in range(rollouts):
                # Rollout collection must run under no_grad — matches skrl's own
                # SequentialTrainer.train(), which wraps act()/env.step()/record_transition()
                # together and only calls post_interaction() (which triggers update()) outside
                # it. Without this, the hidden states act() returns (and record_transition()
                # stores into memory via _rnn_final_states) carry a live, ever-growing autograd
                # graph across the whole rollout. update()'s multi-epoch/multi-minibatch loop
                # then reuses that same stored initial hidden state as input to more than one
                # .backward() call (once per epoch, since sampled_batches is computed once
                # before the epoch loop) — the second one crashes with "Trying to backward
                # through the graph a second time", since the first backward() already freed
                # those upstream (rollout-time) graph nodes.
                with torch.no_grad():
                    actions, outputs = agent.act(observations, env.state(), timestep=timestep, timesteps=timesteps)

                    next_observations, rewards, terminated, truncated, infos = env.step(actions)

                    agent.record_transition(
                        observations=observations,
                        states=env.state(),
                        actions=actions,
                        rewards=rewards,
                        next_observations=next_observations,
                        next_states=env.state(),
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos,
                        timestep=timestep,
                        timesteps=timesteps,
                    )

                actions_history.append(actions.detach().clone())
                if "rnn" in outputs and outputs["rnn"]:
                    hidden_history.append(outputs["rnn"][0].detach().clone())

                done_mask = (terminated | truncated).view(-1)
                reset_ids = done_mask.nonzero(as_tuple=False).view(-1)
                if reset_ids.numel():
                    last_reset_env_ids = reset_ids
                    if agent._rnn and agent._rnn_final_states["policy"]:
                        last_hidden_after_reset = agent._rnn_final_states["policy"][0].detach().clone()

                agent.post_interaction(timestep=timestep, timesteps=timesteps)
                observations = next_observations
                timestep += 1

            # ── per-iteration checks ──
            check_losses(agent.tracking_data, iteration + 1, report)
            check_action_distribution(actions_history, iteration + 1, report)
            check_hidden_state(hidden_history, env.num_envs, iteration + 1, report)
            check_values(agent.memory.get_tensor_by_name("values").detach().clone(), iteration + 1, report)
            if last_hidden_after_reset is not None:
                check_hidden_reset(last_hidden_after_reset, last_reset_env_ids, iteration + 1, report)

            check_gradient_flow(actor, f"actor_iter{iteration + 1}", report)
            check_gradient_flow(critic, f"critic_iter{iteration + 1}", report)

            actor_after = snapshot_params(actor)
            critic_after = snapshot_params(critic)
            check_param_update(actor_before, actor_after, f"actor_iter{iteration + 1}", report)
            check_param_update(critic_before, critic_after, f"critic_iter{iteration + 1}", report)

    except Exception as e:
        report.add(
            "training_loop", False,
            f"Crashed during rollout/training at timestep {timestep}: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()[-1500:]}"
        )

    report.print_summary()
    env.close()
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
