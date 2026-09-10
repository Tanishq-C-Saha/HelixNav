# Copyright (c) 2026, HelixNav Project
# SPDX-License-Identifier: BSD-3-Clause

"""
HelixNav Pre-Flight Environment Sanity Check
=============================================

Runs a systematic health check on the RL environment BEFORE committing
GPU-hours to training. Catches the bugs that silently ruin runs:
  - observation shape/dtype/range mismatches vs. policy expectations
  - NaN / Inf in observations, rewards, or actions
  - stale-observation-after-reset (the CP3.6 class of bug)
  - unbounded rewards that will blow up the value function
  - termination/truncation logic errors
  - action space clipping behaviour
  - VRAM usage under target batch size
  - multi-step rollout stability

Usage:
    # from helix_nav repo root, using Isaac Lab python:
    python scripts/preflight_check.py --task HelixNav-CP7-v0 --num_envs 64
    python scripts/preflight_check.py --task HelixNav-CP7-v0 --num_envs 2048 --full

Exit codes:
    0 = all checks passed
    1 = at least one FAIL
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HelixNav pre-flight environment sanity check.")
parser.add_argument("--task", type=str, required=True, help="Registered gym task ID.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs (64 for quick, 2048 for full).")
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--rollout_steps", type=int, default=200, help="Steps for multi-step rollout check.")
parser.add_argument("--full", action="store_true", help="Run extended checks (more steps, VRAM profiling).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below runs after sim app is up."""

import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import helix_nav.tasks  # noqa: F401 — triggers gym.register

from helix_nav.tasks.manager_based.navigation.models.skrl_policy import DEPTH_FLAT, SCALAR_SIZES, TOTAL_FLAT


# ════════════════════════════════════════════════
#  Expected observation contract (from skrl_policy.py)
# ════════════════════════════════════════════════

# ObservationsCfg.policy uses concatenate_terms=True, so the env hands us one flat
# (N, TOTAL_FLAT) tensor per step. Slice order = field declaration order in
# ObservationsCfg.policy (see manager_configs/observations.py) = the order below.
# Sizes come straight from skrl_policy.py so this can't silently drift out of sync
# with what the policy actually slices.
OBS_LAYOUT = [
    ("depth_images", DEPTH_FLAT, (0.0, 1.0)),  # normalized distance-to-camera, clamped to [0, 1]
    ("lookahead_vectors", SCALAR_SIZES[0], (-5.0, 5.0)),
    ("snap_flags", SCALAR_SIZES[1], (-0.5, 1.5)),
    ("base_velocity", SCALAR_SIZES[2], (-15.0, 15.0)),
    ("prev_actions", SCALAR_SIZES[3], (-5.0, 5.0)),
    ("relative_goal", SCALAR_SIZES[4], (-50.0, 50.0)),
]
assert sum(size for _, size, _ in OBS_LAYOUT) == TOTAL_FLAT, "OBS_LAYOUT sizes don't sum to TOTAL_FLAT"


# ════════════════════════════════════════════════
#  Observation helpers
# ════════════════════════════════════════════════

def get_policy_obs(obs):
    """Return the flat (N, TOTAL_FLAT) policy observation tensor.

    ManagerBasedRLEnv returns observations grouped by observation group,
    e.g. {"policy": tensor}. Unwrap that here so every check works with the
    flat tensor directly.
    """
    if not isinstance(obs, dict):
        raise TypeError(f"Expected observation dict, got {type(obs).__name__}")

    if "policy" not in obs:
        raise KeyError(f"Expected a 'policy' observation group, got keys: {list(obs.keys())}")

    return obs["policy"]


def slice_obs_layout(flat_obs):
    """Split a flat (N, TOTAL_FLAT) tensor into named slices per OBS_LAYOUT.

    Returns an ordered dict of name -> (N, size) tensor. Used purely for
    readable diagnostics — the policy itself only cares about the depth/scalar
    split (see skrl_policy._split_observations).
    """
    slices = {}
    offset = 0
    for name, size, _ in OBS_LAYOUT:
        slices[name] = flat_obs[:, offset : offset + size]
        offset += size
    return slices


# ════════════════════════════════════════════════
#  Result tracking
# ════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    warn: bool = False  # passed but suspicious


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
        print("  PREFLIGHT CHECK SUMMARY")
        print("=" * 72)

        # group by status
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
#  Individual checks
# ════════════════════════════════════════════════

def check_env_creation(task, num_envs, seed, device, report):
    """Check 1: Can the environment be created without error?"""
    try:
        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
        env_cfg.seed = seed

        env = gym.make(task, cfg=env_cfg)
        report.add("env_creation", True, f"Created {num_envs} envs on {device}")
        return env
    except Exception as e:
        report.add("env_creation", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}")
        return None


def check_observation_space(env, report):
    """Check 2: Does the observation space match what the policy expects?"""
    obs_space = env.observation_space

    # must be a flat Box space (concatenate_terms=True)
    if not isinstance(obs_space, gym.spaces.Box):
        report.add(
            "obs_space_type", False,
            f"Expected gym.spaces.Box, got {type(obs_space).__name__}.\n"
            f"Set concatenate_terms=True in ObservationsCfg.policy."
        )
        return

    report.add("obs_space_type", True, "Flat Box observation space (concatenate_terms=True)")

    if obs_space.shape != (TOTAL_FLAT,):
        report.add(
            "obs_shape", False,
            f"Expected shape ({TOTAL_FLAT},) [depth {DEPTH_FLAT} + scalars {sum(SCALAR_SIZES)}], "
            f"got {obs_space.shape}.\n"
            f"Check ObservationsCfg.policy field order/sizes match skrl_policy.py's OBS_LAYOUT."
        )
    else:
        report.add("obs_shape", True, f"Flat obs shape {obs_space.shape} matches policy contract")


def check_action_space(env, report):
    """Check 3: Action space shape and bounds."""
    act_space = env.action_space
    expected_dim = 3  # vx, vy, heading

    if not isinstance(act_space, gym.spaces.Box):
        report.add("action_space", False, f"Expected Box, got {type(act_space).__name__}")
        return

    if act_space.shape != (expected_dim,):
        report.add("action_space", False, f"Expected shape ({expected_dim},), got {act_space.shape}")
        return

    low, high = act_space.low, act_space.high
    report.add(
        "action_space", True,
        f"Box({expected_dim},) bounds: low={low.tolist()}, high={high.tolist()}"
    )


def _check_obs_tensor(tensor, issues, source_label):
    """Shared NaN/Inf/range checks for a single flat obs tensor. Mutates `issues` in place."""
    if tensor.shape[-1] != TOTAL_FLAT:
        issues.append(f"{source_label}: expected last dim {TOTAL_FLAT}, got {tensor.shape[-1]}")
        return

    for name, slice_tensor in slice_obs_layout(tensor).items():
        nan_count = torch.isnan(slice_tensor).sum().item()
        inf_count = torch.isinf(slice_tensor).sum().item()

        if nan_count > 0:
            issues.append(f"{source_label} obs['{name}']: {nan_count} NaN values")
        if inf_count > 0:
            issues.append(f"{source_label} obs['{name}']: {inf_count} Inf values")

    return


def check_reset_observations(env, report):
    """Check 4: Reset produces valid observations (no NaN/Inf, correct shape, within range)."""
    obs, info = env.reset()

    issues = []

    try:
        flat_obs = get_policy_obs(obs)
    except (TypeError, KeyError) as e:
        report.add("reset_observations", False, str(e))
        return obs

    if flat_obs.shape[-1] != TOTAL_FLAT:
        issues.append(f"Expected flat obs last dim {TOTAL_FLAT}, got {flat_obs.shape[-1]}")
    else:
        for name, slice_tensor in slice_obs_layout(flat_obs).items():
            nan_count = torch.isnan(slice_tensor).sum().item()
            inf_count = torch.isinf(slice_tensor).sum().item()

            if nan_count > 0:
                issues.append(f"{name}: {nan_count} NaN values at reset")
            if inf_count > 0:
                issues.append(f"{name}: {inf_count} Inf values at reset")

            lo, hi = next(r for n, _, r in OBS_LAYOUT if n == name)
            out_of_range = ((slice_tensor < lo) | (slice_tensor > hi)).sum().item()
            total = slice_tensor.numel()
            if out_of_range > 0:
                vmin, vmax = slice_tensor.min().item(), slice_tensor.max().item()
                pct = 100.0 * out_of_range / total
                issues.append(
                    f"{name}: {out_of_range}/{total} ({pct:.1f}%) values outside "
                    f"[{lo}, {hi}], actual range [{vmin:.3f}, {vmax:.3f}]"
                )

    if issues:
        report.add("reset_observations", False, "\n".join(issues))
    else:
        report.add("reset_observations", True, "All obs slices finite and within expected ranges after reset")

    return obs


def check_step_basic(env, report):
    """Check 5: A single step with zero actions doesn't crash and produces valid data."""
    num_envs = env.scene.num_envs
    device = env.device

    zero_actions = torch.zeros(num_envs, 3, device=device)

    try:
        obs, reward, terminated, truncated, info = env.step(zero_actions)
    except Exception as e:
        report.add("step_zero_action", False, f"Crashed on zero-action step: {e}")
        return None

    issues = []

    # reward checks
    if torch.isnan(reward).any():
        issues.append(f"NaN in reward: {torch.isnan(reward).sum().item()} envs")
    if torch.isinf(reward).any():
        issues.append(f"Inf in reward: {torch.isinf(reward).sum().item()} envs")

    rmin, rmax = reward.min().item(), reward.max().item()
    if abs(rmax) > 1000 or abs(rmin) > 1000:
        issues.append(f"Reward magnitude suspicious: [{rmin:.2f}, {rmax:.2f}]")

    # terminated/truncated dtype
    if terminated.dtype != torch.bool:
        issues.append(f"terminated dtype: {terminated.dtype} (expected bool)")
    if truncated.dtype != torch.bool:
        issues.append(f"truncated dtype: {truncated.dtype} (expected bool)")

    # obs validity after step
    try:
        flat_obs = get_policy_obs(obs)
        _check_obs_tensor(flat_obs, issues, "post-step")
    except (TypeError, KeyError) as e:
        issues.append(str(e))

    if issues:
        report.add("step_zero_action", False, "\n".join(issues))
    else:
        report.add(
            "step_zero_action", True,
            f"Zero-action step OK. Reward range: [{rmin:.4f}, {rmax:.4f}]"
        )

    return obs, reward


def check_random_actions(env, report):
    """Check 6: Random actions for a few steps — catch action clipping issues."""
    num_envs = env.num_envs
    device = env.device
    n_steps = 20
    issues = []

    reward_accum = []

    for step in range(n_steps):
        # sample random actions in [-1, 1]
        actions = torch.rand(num_envs, 3, device=device) * 2 - 1

        try:
            obs, reward, terminated, truncated, info = env.step(actions)
        except Exception as e:
            issues.append(f"Crashed at random-action step {step}: {e}")
            break

        if torch.isnan(reward).any():
            issues.append(f"Step {step}: NaN reward ({torch.isnan(reward).sum().item()} envs)")
        if torch.isinf(reward).any():
            issues.append(f"Step {step}: Inf reward ({torch.isinf(reward).sum().item()} envs)")

        try:
            flat_obs = get_policy_obs(obs)
            if torch.isnan(flat_obs).any():
                issues.append(f"Step {step}: NaN in observations")
        except (TypeError, KeyError) as e:
            issues.append(f"Step {step}: {e}")
            break

        reward_accum.append(reward.clone())

    if not issues:
        all_rewards = torch.stack(reward_accum)  # (steps, num_envs)
        rmin = all_rewards.min().item()
        rmax = all_rewards.max().item()
        rmean = all_rewards.mean().item()
        report.add(
            "random_actions", True,
            f"{n_steps} random-action steps OK.\n"
            f"Reward: min={rmin:.4f}, max={rmax:.4f}, mean={rmean:.4f}"
        )
    else:
        report.add("random_actions", False, "\n".join(issues[:10]))


def check_reset_consistency(env, report):
    """Check 7: After reset, observations are from the NEW episode, not stale.

    This is the CP3.6 bug class — stale obs after reset.
    Strategy: reset, record obs, step a bunch, reset again, check obs changed.
    """
    obs1, _ = env.reset()

    # run some steps to move state away from initial
    for _ in range(30):
        actions = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
        env.step(actions)

    # reset again
    obs2, _ = env.reset()

    issues = []

    try:
        flat_obs1 = get_policy_obs(obs1)
        flat_obs2 = get_policy_obs(obs2)
    except (TypeError, KeyError) as e:
        report.add("reset_consistency", False, str(e))
        return

    slices1 = slice_obs_layout(flat_obs1)
    slices2 = slice_obs_layout(flat_obs2)

    for name in slices1:
        t1, t2 = slices1[name], slices2[name]

        # for randomised envs, obs should differ across resets
        # (unless seed forces identical maps — check a few envs)
        n_check = min(8, env.num_envs)
        identical_count = 0
        for i in range(n_check):
            if torch.allclose(t1[i], t2[i], atol=1e-6):
                identical_count += 1

        if identical_count == n_check and name != "prev_actions":
            # prev_actions is zero after both resets, so always identical — skip
            issues.append(
                f"obs['{name}']: all {n_check} sampled envs identical across resets.\n"
                f"  If randomisation is on, this suggests stale observations."
            )

    if issues:
        report.add("reset_consistency", False, "\n".join(issues))
    else:
        report.add(
            "reset_consistency", True,
            "Observations differ across resets (randomisation producing fresh episodes)"
        )


def check_episode_termination(env, report):
    """Check 8: Episodes actually terminate within reasonable time."""
    env.reset()

    terminated_any = False
    truncated_any = False
    max_steps = 800  # 80 seconds at 10Hz — well beyond episode_length_s=60

    for step in range(max_steps):
        actions = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
        obs, reward, terminated, truncated, info = env.step(actions)

        if terminated.any():
            terminated_any = True
        if truncated.any():
            truncated_any = True

        if terminated_any and truncated_any:
            break

    detail_parts = []
    if terminated_any:
        detail_parts.append(f"Termination observed within {step + 1} steps")
    else:
        detail_parts.append(f"No termination in {max_steps} steps")
    if truncated_any:
        detail_parts.append(f"Truncation (time limit) observed within {step + 1} steps")
    else:
        detail_parts.append(f"No truncation in {max_steps} steps")

    # truncation must happen (episode_length_s = 60, dt=0.1s → 600 steps)
    if not truncated_any:
        report.add("episode_termination", False, "\n".join(detail_parts))
    else:
        report.add("episode_termination", True, "\n".join(detail_parts))


def check_reward_boundedness(env, report):
    """Check 9: Extended rollout to verify rewards stay bounded (no CP6.5 NaN explosions)."""
    env.reset()
    n_steps = args_cli.rollout_steps if not args_cli.full else 600

    all_rewards = []
    nan_step = None

    for step in range(n_steps):
        actions = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
        obs, reward, terminated, truncated, info = env.step(actions)

        if torch.isnan(reward).any() or torch.isinf(reward).any():
            nan_step = step
            break

        all_rewards.append(reward.clone())

    if nan_step is not None:
        report.add(
            "reward_bounded", False,
            f"NaN/Inf reward at step {nan_step} — CP6.5-class unbounded reward bug"
        )
        return

    rewards = torch.stack(all_rewards)  # (steps, num_envs)
    rmin = rewards.min().item()
    rmax = rewards.max().item()
    rmean = rewards.mean().item()
    rstd = rewards.std().item()

    # check for suspicious patterns
    warn = False
    detail = f"{n_steps}-step rollout. Reward stats:\n  min={rmin:.4f}  max={rmax:.4f}  mean={rmean:.4f}  std={rstd:.4f}"

    if abs(rmax) > 200 or abs(rmin) > 200:
        detail += f"\n  ⚠️  Large reward magnitude — check PBRS scaling vs goal bonus"
        warn = True

    if rstd < 1e-6:
        detail += f"\n  ⚠️  Zero reward variance — reward function may be constant"
        warn = True

    # check for monotonic reward drift (sign of unbounded accumulation)
    per_step_mean = rewards.mean(dim=1)  # (steps,)
    if len(per_step_mean) > 50:
        first_half = per_step_mean[:len(per_step_mean) // 2].mean().item()
        second_half = per_step_mean[len(per_step_mean) // 2:].mean().item()
        if abs(second_half) > 10 * max(abs(first_half), 0.01):
            detail += f"\n  ⚠️  Reward drifting: first_half_mean={first_half:.4f}, second_half_mean={second_half:.4f}"
            warn = True

    report.add("reward_bounded", True, detail, warn=warn)


def check_vram_usage(env, report):
    """Check 10: Report VRAM usage with current env count."""
    if not torch.cuda.is_available():
        report.add("vram_usage", True, "No CUDA — skipped", warn=True)
        return

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - reserved

    detail = (
        f"VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, "
        f"{free:.2f} GB free of {total:.2f} GB total\n"
        f"  Headroom for policy + optimizer: ~{free:.1f} GB"
    )

    warn = free < 2.0  # less than 2GB free is tight for training
    if warn:
        detail += "\n  ⚠️  Low VRAM headroom — training may OOM. Reduce num_envs or sequence_length."

    report.add("vram_usage", True, detail, warn=warn)


def check_obs_normalization_ranges(env, report):
    """Check 11: Run a short rollout and report actual observation ranges.

    Helps catch observations that need normalization before they reach the network.
    """
    env.reset()
    n_steps = 100

    obs_stats = {}

    for step in range(n_steps):
        actions = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
        obs, *_ = env.step(actions)

        flat_obs = get_policy_obs(obs)
        for name, tensor in slice_obs_layout(flat_obs).items():
            if name not in obs_stats:
                obs_stats[name] = {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0}
            s = obs_stats[name]
            s["min"] = min(s["min"], tensor.min().item())
            s["max"] = max(s["max"], tensor.max().item())
            s["sum"] += tensor.float().mean().item()
            s["count"] += 1

    detail_lines = [f"Observation ranges over {n_steps} steps:"]
    warn = False

    for name, s in obs_stats.items():
        mean = s["sum"] / max(s["count"], 1)
        line = f"  {name:25s}  min={s['min']:9.3f}  max={s['max']:9.3f}  mean={mean:9.3f}"

        # flag if range is extreme (network inputs ideally in [-10, 10])
        if abs(s["min"]) > 50 or abs(s["max"]) > 50:
            line += "  ← LARGE"
            warn = True

        detail_lines.append(line)

    report.add("obs_ranges", True, "\n".join(detail_lines), warn=warn)


# ════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════

def main():
    report = Report()

    print("\n" + "=" * 72)
    print(f"  HelixNav Pre-Flight Check: {args_cli.task}")
    print(f"  num_envs={args_cli.num_envs}  seed={args_cli.seed}  device={args_cli.device}")
    print(f"  mode={'FULL' if args_cli.full else 'QUICK'}")
    print("=" * 72 + "\n")

    t0 = time.time()

    # ── 1. Create environment ──
    print("[1/11] Creating environment...")
    env_wrapped = check_env_creation(args_cli.task, args_cli.num_envs, args_cli.seed, args_cli.device, report)
    env = env_wrapped.unwrapped
    if env is None:
        report.print_summary()
        return 1

    # ── 2. Observation space ──
    print("[2/11] Checking observation space...")
    check_observation_space(env, report)

    # ── 3. Action space ──
    print("[3/11] Checking action space...")
    check_action_space(env, report)

    # ── 4. Reset observations ──
    print("[4/11] Checking reset observations...")
    check_reset_observations(env, report)

    # ── 5. Single step ──
    print("[5/11] Checking single zero-action step...")
    check_step_basic(env, report)

    # ── 6. Random actions ──
    print("[6/11] Checking random-action steps...")
    check_random_actions(env, report)

    # ── 7. Reset consistency (stale obs check) ──
    print("[7/11] Checking reset consistency (stale observation detection)...")
    check_reset_consistency(env, report)

    # ── 8. Episode termination ──
    print("[8/11] Checking episode termination/truncation...")
    check_episode_termination(env, report)

    # ── 9. Reward boundedness ──
    print("[9/11] Checking reward boundedness over extended rollout...")
    check_reward_boundedness(env, report)

    # ── 10. VRAM ──
    print("[10/11] Checking VRAM usage...")
    check_vram_usage(env, report)

    # ── 11. Observation ranges ──
    print("[11/11] Profiling observation ranges...")
    check_obs_normalization_ranges(env, report)

    elapsed = time.time() - t0
    print(f"\nPreflight completed in {elapsed:.1f}s")

    # ── Summary ──
    report.print_summary()

    env.close()
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
