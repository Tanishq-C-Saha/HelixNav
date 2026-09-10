# Copyright (c) 2026, HelixNav Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Play a trained HelixNav CP7 checkpoint with skrl's PPO_RNN — custom play script.
===================================================================================

Mirrors train_cp7.py's reasoning: skrl 2.0.0's ``Runner`` (what the stock
scripts/skrl/play.py uses) can't build our custom GRU+CNN models, and stock
play.py additionally calls ``runner.agent.set_running_mode("eval")``, a method
name from an older skrl API that doesn't exist on this skrl version's ``Agent``
(it's ``enable_training_mode`` here). So this script builds the environment and
models by hand — the same pattern as train_cp7.py/training_dryrun.py — loads a
checkpoint, and runs a simple deterministic inference loop.

Unlike training, play deliberately does NOT call ``agent.record_transition()``:
for our asymmetric actor/critic (policy is not value), that method has a real
bug when ``self.training`` is False (exactly play's case) — it skips the block
that defines the local ``outputs`` variable, then unconditionally reaches
``outputs.get("rnn", [])`` a few lines later, raising ``UnboundLocalError``.
record_transition() also isn't needed for play (it only feeds memory/reward
shaping for training) except for one thing it normally does as a side effect:
carrying the GRU hidden state forward between steps and zeroing it on episode
boundaries. This script replicates just that part manually.

Usage:
    python scripts/skrl/play_cp7.py --task HelixNav-CP7-Play-v0
    python scripts/skrl/play_cp7.py --task HelixNav-CP7-Play-v0 --checkpoint /path/to/agent_9000.pt
    python scripts/skrl/play_cp7.py --task HelixNav-CP7-Play-v0 --video --video_length 200
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained HelixNav CP7 checkpoint with skrl PPO_RNN.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the play episode.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint. Defaults to the most recent checkpoint under this task's experiment log directory.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--difficulty",
    type=int,
    default=None,
    help=(
        "Pin the curriculum map difficulty (see CurriculumsCfg.map_difficulty's "
        "thresholds, e.g. 1-3) for the whole play session, overriding curriculum "
        "advancement. Omit to use the curriculum's normal starting difficulty."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help=(
        "Stop after this many completed episodes (summed across all envs) and write a "
        "trajectory plot per episode plus an aggregate metrics dashboard/CSV. Set to 0 "
        "to run indefinitely instead (until the window is closed), like plain play — no "
        "final dashboard is written in that case, but per-episode trajectory plots still are."
    ),
)
parser.add_argument(
    "--plot_dir",
    type=str,
    default=None,
    help="Directory for trajectory plots / metrics dashboard / CSV. Defaults to <experiment log dir>/play_analysis.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import random
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import helix_nav.tasks  # noqa: F401 — triggers gym.register

from helix_nav.tasks.manager_based.navigation.models.skrl_policy import build_models
from helix_nav.tasks.manager_based.navigation.mdp.visualize_utils import (
    compute_spl,
    path_length,
    plot_episode_metrics_dashboard,
    plot_nav_state,
    save_episode_metrics_csv,
)

from skrl.agents.torch.ppo import PPO_RNN, PPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR


# This task registers its agent yaml under "skrl_cfg_entry_point" (see
# config/go2/__init__.py) — same key train_cp7.py resolves.
AGENT_CFG_ENTRY_POINT = "skrl_cfg_entry_point"

# Same string-to-class resolution train_cp7.py's build_ppo_cfg() does — Runner would
# normally do this internally via Runner._component(); we do it by hand since we're
# not using Runner. Kept in sync with train_cp7.py's copy.
_PREPROCESSOR_CLASSES = {
    None: None,
    "RunningStandardScaler": RunningStandardScaler,
}
_SCHEDULER_CLASSES = {
    None: None,
    "KLAdaptiveLR": KLAdaptiveLR,
}
_EXPERIMENT_FIELDS = {
    "directory", "experiment_name", "write_interval", "checkpoint_interval",
    "store_separately", "wandb", "wandb_kwargs",
}


def build_ppo_cfg(raw_agent_cfg: dict, env, log_dir: str) -> PPO_CFG:
    """Convert the raw yaml-loaded agent dict into a PPO_CFG instance.

    See train_cp7.py's copy of this function for the full rationale. Play doesn't
    train, but PPO_RNN still needs a fully-formed cfg (rollouts controls the GRU
    sequence_length passed to build_models, and the preprocessors/experiment fields
    need real types/paths rather than the raw yaml strings).
    """
    cfg_dict = dict(raw_agent_cfg)
    cfg_dict.pop("class", None)

    for key in ("observation_preprocessor", "value_preprocessor", "state_preprocessor"):
        name = cfg_dict.get(key)
        if name not in _PREPROCESSOR_CLASSES:
            raise ValueError(f"Unsupported {key} '{name}' — add it to _PREPROCESSOR_CLASSES in this script.")
        cfg_dict[key] = _PREPROCESSOR_CLASSES[name]

    if cfg_dict.get("observation_preprocessor") is not None:
        cfg_dict["observation_preprocessor_kwargs"] = {
            "size": env.observation_space.shape[0],
            "device": env.device,
        }
    if cfg_dict.get("value_preprocessor") is not None:
        cfg_dict["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

    scheduler_name = cfg_dict.get("learning_rate_scheduler")
    if scheduler_name not in _SCHEDULER_CLASSES:
        raise ValueError(
            f"Unsupported learning_rate_scheduler '{scheduler_name}' — "
            f"add it to _SCHEDULER_CLASSES in this script."
        )
    cfg_dict["learning_rate_scheduler"] = _SCHEDULER_CLASSES[scheduler_name]

    raw_experiment = cfg_dict.get("experiment", {}) or {}
    experiment = {k: v for k, v in raw_experiment.items() if k in _EXPERIMENT_FIELDS}
    experiment["directory"] = os.path.dirname(log_dir)
    experiment["experiment_name"] = os.path.basename(log_dir)
    # play never logs or checkpoints
    experiment["write_interval"] = 0
    experiment["checkpoint_interval"] = 0
    cfg_dict["experiment"] = experiment

    return PPO_CFG(**cfg_dict)


@hydra_task_config(args_cli.task, AGENT_CFG_ENTRY_POINT)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    """Play a trained PPO_RNN checkpoint on HelixNav CP7."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    rollouts = agent_cfg["agent"]["rollouts"]

    # resolve checkpoint (mirrors stock play.py's resolution logic)
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=".*_PPO_RNN_torch", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # ── create + wrap environment ──
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video of play episode.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # ── build models + agent (no memory, no optimizer — play never trains) ──
    models = build_models(env, sequence_length=rollouts)
    cfg = build_ppo_cfg(agent_cfg["agent"], env, log_dir)

    agent = PPO_RNN(
        models=models,
        memory=None,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        cfg=cfg,
    )
    # sets up _rnn_initial_states (zeros) and _rnn_sequence_length, and puts both
    # models in eval mode via enable_models_training_mode(False) — exactly what
    # play wants, and memory=None makes it skip memory tensor creation entirely.
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(False)

    # train_cp7.py replaces agent.optimizer with a deduplicated 3-param-group Adam
    # (see its build_deduplicated_optimizer) before any checkpoint is saved, so the
    # checkpoint's "optimizer" entry has 3 param groups. This agent never built that
    # replacement (play never calls .step(), so there's no reason to) — its default
    # PPO_RNN-constructed optimizer has 1 param group, and Optimizer.load_state_dict()
    # requires an exact param-group-count match. Since play has no use for optimizer
    # state at all, just don't load it — Agent.load() skips any checkpoint entry
    # that isn't in checkpoint_modules (with a warning) rather than erroring.
    del agent.checkpoint_modules["optimizer"]

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)

    # ── episode tracking (trajectories + navigation-RL evaluation metrics) ──
    plot_dir = args_cli.plot_dir or os.path.join(log_dir, "play_analysis")
    num_envs = env.num_envs
    raw_env = env.unwrapped

    trajectories_local = [[] for _ in range(num_envs)]  # (x, y) local coords, recorded each step
    episode_reward = torch.zeros(num_envs, device=env.device)
    episode_steps = [0] * num_envs
    episode_context = [None] * num_envs  # snapshotted at each episode's start (see below)
    episode_metrics = []
    episode_counter = 0

    def snapshot_episode_context(env_id: int) -> dict:
        """Capture the map/path/goal an episode is using, right as it starts.

        mdp.events.reset_map_and_spawn regenerates _map_specs/_paths_local/etc. for an env
        on every reset, so by the time a LATER reset happens for that same env (i.e. this
        episode has ended), those attributes already reflect the NEXT episode. Snapshotting
        here — right after each reset — is what lets episode-end handling below still refer
        to the map THIS (now-finished) episode was actually played on.
        """
        path_len = int(raw_env._path_lengths[env_id].item())
        path_local = raw_env._paths_local[env_id, :path_len].cpu().numpy() if path_len > 0 else None
        map_spec = raw_env._map_specs[env_id]
        return {
            "difficulty": int(raw_env._current_difficulty),
            "path_local": path_local,
            "start_local": raw_env._start_positions_local[env_id].cpu().numpy(),
            "goal_local": raw_env._goal_positions_local[env_id].cpu().numpy(),
            "occupancy_grid": map_spec.occupancy_grid if map_spec is not None else None,
        }

    # ── inference loop ──
    observations, _ = env.reset()

    if args_cli.difficulty is not None:
        # env._current_difficulty lives on the raw ManagerBasedRLEnv, not the skrl
        # wrapper — env.unwrapped resolves through it via skrl's Wrapper.__getattr__
        # proxy (gym's own .unwrapped protocol), but writes must target that raw
        # object directly (assignment doesn't go through the same proxy as reads).
        #
        # The reset just above is also the FIRST reset ever for this env, which
        # lazily runs mdp.events.init_nav_state() and unconditionally hardcodes
        # _current_difficulty = 1 — clobbering any pin set before it. So pin now,
        # then reset again: this second reset's map generation (mdp.events.
        # reset_map_and_spawn) reads the now-pinned value, so even the first
        # visible episode is at the requested difficulty instead of 1.
        env.unwrapped._current_difficulty = args_cli.difficulty
        observations, _ = env.reset()

    for env_id in range(num_envs):
        episode_context[env_id] = snapshot_episode_context(env_id)

    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        if args_cli.difficulty is not None:
            # CurriculumsCfg.map_difficulty can still advance _current_difficulty
            # on any reset that happens inside the upcoming env.step() (for envs
            # whose episode just ended) — re-assert the pin right before every
            # step so no reset, for the rest of this play session, ever sees a
            # difficulty other than the one requested.
            env.unwrapped._current_difficulty = args_cli.difficulty

        # Record each env's current position for the trajectory plot. This is the
        # PRE-step position (state the upcoming action is taken from) rather than the
        # post-step one: Isaac Lab resets a finished env's pose internally, inside
        # env.step() itself, so by the time step() returns for an env whose episode
        # just ended, root_pos_w is already the NEXT episode's spawn pose, not where
        # this episode actually ended. Recording pre-step means the very last ~0.1s
        # of a trajectory (the final transition into the goal/collision) isn't drawn
        # — a minor, deliberate approximation rather than plotting the wrong point.
        robot_pos_local = (
            raw_env.scene["robot"].data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2]
        ).cpu().numpy()
        for env_id in range(num_envs):
            trajectories_local[env_id].append(robot_pos_local[env_id])

        with torch.inference_mode():
            actions, outputs = agent.act(observations, env.state(), timestep=0, timesteps=0)
            # deterministic (mean) actions for play, not stochastic samples
            actions = outputs.get("mean_actions", actions)

            next_observations, rewards, terminated, truncated, infos = env.step(actions)

            episode_reward += rewards.view(-1)
            for env_id in range(num_envs):
                episode_steps[env_id] += 1

            finished = (terminated | truncated).nonzero(as_tuple=False)

            # PPO_RNN.act() writes the new hidden state into _rnn_final_states but only
            # record_transition() normally copies it forward into _rnn_initial_states for
            # the next step (and zeroes it on episode boundaries) — replicate just that
            # here, since record_transition() itself isn't safe to call in eval mode for
            # our asymmetric policy/value models (see module docstring).
            if agent._rnn:
                agent._rnn_initial_states["policy"] = agent._rnn_final_states["policy"]
                if finished.numel():
                    for hidden_state in agent._rnn_initial_states["policy"]:
                        hidden_state[:, finished[:, 0]] = 0

        observations = next_observations

        if finished.numel():
            goal_reached = raw_env.termination_manager.get_term("goal_reached")
            collided = raw_env.termination_manager.get_term("collisions")

            for env_id in finished[:, 0].tolist():
                ctx = episode_context[env_id]
                trajectory = np.array(trajectories_local[env_id]) if trajectories_local[env_id] else None

                success = bool(goal_reached[env_id].item())
                did_collide = bool(collided[env_id].item())
                timed_out = bool(truncated[env_id].item()) and not success and not did_collide

                actual_len = path_length(trajectory)
                optimal_len = path_length(ctx["path_local"])
                spl = compute_spl(success, actual_len, optimal_len)

                episode_counter += 1
                outcome = "SUCCESS" if success else ("COLLISION" if did_collide else "TIMEOUT")
                print(
                    f"[episode {episode_counter}] env={env_id} D{ctx['difficulty']} {outcome} "
                    f"steps={episode_steps[env_id]} SPL={spl:.3f} reward={episode_reward[env_id].item():.2f}"
                )

                episode_metrics.append({
                    "episode": episode_counter,
                    "env_id": env_id,
                    "difficulty": ctx["difficulty"],
                    "success": success,
                    "collided": did_collide,
                    "timed_out": timed_out,
                    "steps": episode_steps[env_id],
                    "duration_s": round(episode_steps[env_id] * dt, 3),
                    "path_length_actual": round(actual_len, 3),
                    "path_length_optimal": round(optimal_len, 3),
                    "spl": round(spl, 4),
                    "cumulative_reward": round(episode_reward[env_id].item(), 4),
                })

                if ctx["occupancy_grid"] is not None:
                    plot_nav_state(
                        occupancy_grid=ctx["occupancy_grid"],
                        path_world=ctx["path_local"],
                        start_world=ctx["start_local"],
                        goal_world=ctx["goal_local"],
                        trajectory_world=trajectory,
                        title=(
                            f"Play | Env {env_id} | D{ctx['difficulty']} | Episode {episode_counter} | "
                            f"{outcome} | SPL={spl:.2f}"
                        ),
                        save_path=os.path.join(
                            plot_dir, "trajectories", f"env_{env_id}", f"ep_{episode_counter:04d}.jpg"
                        ),
                    )

                # this env has already respawned internally (inside the env.step() call
                # above) — start tracking its new episode fresh.
                trajectories_local[env_id] = []
                episode_reward[env_id] = 0.0
                episode_steps[env_id] = 0
                episode_context[env_id] = snapshot_episode_context(env_id)

            if args_cli.num_episodes and episode_counter >= args_cli.num_episodes:
                break

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if episode_metrics:
        plot_episode_metrics_dashboard(
            episode_metrics,
            title=f"HelixNav CP7 Play — {os.path.basename(resume_path)}",
            save_path=os.path.join(plot_dir, "metrics_summary.jpg"),
        )
        save_episode_metrics_csv(episode_metrics, os.path.join(plot_dir, "episode_metrics.csv"))

        n = len(episode_metrics)
        success_rate = 100.0 * sum(m["success"] for m in episode_metrics) / n
        collision_rate = 100.0 * sum(m["collided"] for m in episode_metrics) / n
        timeout_rate = 100.0 * sum(m["timed_out"] for m in episode_metrics) / n
        mean_spl = sum(m["spl"] for m in episode_metrics) / n
        mean_duration = sum(m["duration_s"] for m in episode_metrics) / n
        mean_actual = sum(m["path_length_actual"] for m in episode_metrics) / n
        mean_optimal = sum(m["path_length_optimal"] for m in episode_metrics) / n
        mean_reward = sum(m["cumulative_reward"] for m in episode_metrics) / n

        print("\n" + "=" * 60)
        print(f"  PLAY SUMMARY — {n} episodes")
        print("=" * 60)
        print(f"  Success rate:            {success_rate:.1f}%")
        print(f"  Collision rate:          {collision_rate:.1f}%")
        print(f"  Timeout rate:            {timeout_rate:.1f}%")
        print(f"  Mean SPL:                {mean_spl:.3f}")
        print(f"  Mean episode duration:   {mean_duration:.1f} s")
        print(f"  Mean distance travelled: {mean_actual:.2f} m  (A* optimal: {mean_optimal:.2f} m)")
        print(f"  Mean cumulative reward:  {mean_reward:.2f}")
        print(f"  Saved trajectory plots, metrics_summary.jpg, and episode_metrics.csv to: {plot_dir}")
        print("=" * 60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
