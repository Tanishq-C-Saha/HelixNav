# Copyright (c) 2026, HelixNav Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Train HelixNav CP7 with skrl's PPO_RNN — custom training script.
==================================================================

skrl 2.0.0's ``Runner`` cannot be used for this task: its model builder only
produces generic MLPs from a ``network: layers: [...]`` spec (no hook for a
custom class like ``HelixNavActorRNN``), and its agent whitelist has no RNN
variants at all (``ppo_rnn``/``rpo_rnn`` aren't in it). So this script builds
the environment, models, memory, and agent by hand — the same pattern as
scripts/skrl/training_dryrun.py — and hands them to skrl's ``SequentialTrainer``
for the actual training loop (which gives TensorBoard logging, checkpointing,
and progress tracking without us reimplementing them).

The agent hyperparameters still come from config/go2/agents/skrl_ppo_cfg.yaml
(registered as this task's "skrl_cfg_entry_point"), loaded the same way
train.py loads it — just converted into a PPO_CFG object ourselves instead of
handing the raw dict to Runner.

Usage:
    python scripts/skrl/train_cp7.py --task HelixNav-CP7-v0 --num_envs 2048
    python scripts/skrl/train_cp7.py --task HelixNav-CP7-v0 --num_envs 2048 --max_iterations 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train HelixNav CP7 with skrl PPO_RNN.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations (rollouts).")
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
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import helix_nav.tasks  # noqa: F401 — triggers gym.register

from helix_nav.tasks.manager_based.navigation.models.skrl_policy import build_models

from skrl.agents.torch.ppo import PPO_RNN, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer


# This task registers its agent yaml under "skrl_cfg_entry_point" (see
# config/go2/__init__.py) — same key train.py resolves for algorithm="ppo".
AGENT_CFG_ENTRY_POINT = "skrl_cfg_entry_point"

# name -> class for the string-valued fields skrl's own Runner would normally resolve
# via _component(). Extend this if the yaml starts using more than these.
_PREPROCESSOR_CLASSES = {
    None: None,
    "RunningStandardScaler": RunningStandardScaler,
}
_SCHEDULER_CLASSES = {
    None: None,
    "KLAdaptiveLR": KLAdaptiveLR,
}

# Fields PPO_CFG's nested ExperimentCfg dataclass actually supports in this skrl
# version — anything else in the yaml's `experiment:` block (e.g. `mlflow`) is dropped.
_EXPERIMENT_FIELDS = {
    "directory", "experiment_name", "write_interval", "checkpoint_interval",
    "store_separately", "wandb", "wandb_kwargs",
}


def build_ppo_cfg(raw_agent_cfg: dict, env, log_dir: str) -> PPO_CFG:
    """Convert the raw yaml-loaded agent dict into a PPO_CFG instance.

    Runner normally does this string-to-class resolution internally via
    Runner._component() before handing the result to PPO_CFG(**cfg); we do the
    same conversion by hand since we're not using Runner.
    """
    cfg_dict = dict(raw_agent_cfg)
    cfg_dict.pop("class", None)  # selects the agent class in Runner's cfg; not a PPO_CFG field

    # observation/value/state preprocessor: string name -> class
    for key in ("observation_preprocessor", "value_preprocessor", "state_preprocessor"):
        name = cfg_dict.get(key)
        if name not in _PREPROCESSOR_CLASSES:
            raise ValueError(f"Unsupported {key} '{name}' — add it to _PREPROCESSOR_CLASSES in this script.")
        cfg_dict[key] = _PREPROCESSOR_CLASSES[name]

    # size/device in the yaml are placeholders — compute them from the live env instead
    # of trusting hardcoded values that can drift out of sync with the obs contract.
    if cfg_dict.get("observation_preprocessor") is not None:
        cfg_dict["observation_preprocessor_kwargs"] = {
            "size": env.observation_space.shape[0],
            "device": env.device,
        }
    if cfg_dict.get("value_preprocessor") is not None:
        cfg_dict["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

    # learning rate scheduler: string name -> class
    scheduler_name = cfg_dict.get("learning_rate_scheduler")
    if scheduler_name not in _SCHEDULER_CLASSES:
        raise ValueError(
            f"Unsupported learning_rate_scheduler '{scheduler_name}' — "
            f"add it to _SCHEDULER_CLASSES in this script."
        )
    cfg_dict["learning_rate_scheduler"] = _SCHEDULER_CLASSES[scheduler_name]

    # experiment: drop unsupported keys (e.g. yaml's aspirational "mlflow"), then
    # point directory/experiment_name at this run's actual log directory.
    raw_experiment = cfg_dict.get("experiment", {}) or {}
    experiment = {k: v for k, v in raw_experiment.items() if k in _EXPERIMENT_FIELDS}
    experiment["directory"] = os.path.dirname(log_dir)
    experiment["experiment_name"] = os.path.basename(log_dir)
    cfg_dict["experiment"] = experiment

    return PPO_CFG(**cfg_dict)


def build_deduplicated_optimizer(actor, critic, learning_rate):
    """Build an Adam optimizer that lists the shared encoder's params exactly once.

    PPO_RNN builds its own optimizer internally as
    ``Adam(chain(policy.parameters(), value.parameters()))`` whenever
    ``policy is not value``. Since our actor/critic share one ``encoder``
    submodule, that optimizer double-lists every shared-encoder parameter —
    PyTorch's duplicate guard only checks across separate param_groups, not
    within one list, so this goes undetected and ``optimizer.step()`` applies
    the Adam update twice per step to those params (doubling their effective
    learning rate and corrupting momentum). See training_dryrun.py's
    check_optimizer_param_duplication for the check that caught this.
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


@hydra_task_config(args_cli.task, AGENT_CFG_ENTRY_POINT)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    """Train HelixNav CP7 with a manually-wired PPO_RNN + SequentialTrainer."""
    # override configurations with non-hydra CLI arguments (mirrors train.py)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    rollouts = agent_cfg["agent"]["rollouts"]
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # logging directory (same layout as train.py)
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_PPO_RNN_torch"
    print(f"Exact experiment name requested from command line: {log_dir_name}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    log_dir = os.path.join(log_root_path, log_dir_name)

    env_cfg.log_dir = log_dir

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # ── create + wrap environment ──
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # ── build models (shared depth-CNN + GRU encoder) ──
    models = build_models(env, sequence_length=rollouts)
    actor, critic = models["policy"], models["value"]

    # ── build agent cfg + memory + PPO_RNN agent ──
    cfg = build_ppo_cfg(agent_cfg["agent"], env, log_dir)
    # capture as a plain float now — PPO_CFG.expand() (called inside PPO_RNN's __init__
    # below, via Agent.__init__) turns this into a (policy_lr, value_lr) tuple in place.
    # (if the yaml ever sets a [policy_lr, value_lr] pair instead of one scalar, our
    # single deduplicated optimizer only supports one shared rate — use the first.)
    learning_rate = cfg.learning_rate
    if isinstance(learning_rate, (list, tuple)):
        learning_rate = learning_rate[0]
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    agent = PPO_RNN(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        cfg=cfg,
    )

    # Replace PPO_RNN's default (duplicate-listing) optimizer before SequentialTrainer
    # calls agent.init() — do NOT call agent.init() ourselves here: SequentialTrainer's
    # __init__ does that with the real trainer cfg (timesteps etc). Calling it twice
    # would permanently disable "auto" write_interval/checkpoint_interval — the first
    # call resolves "auto" using whatever trainer_cfg it's given, and Agent.init() only
    # re-resolves it while it's still literally the string "auto".
    agent.optimizer = build_deduplicated_optimizer(actor, critic, learning_rate)
    agent.checkpoint_modules["optimizer"] = agent.optimizer
    neutralize_shared_encoder_training_toggle(critic)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    trainer_cfg = {k: v for k, v in agent_cfg["trainer"].items() if k != "class"}
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)

    trainer.train()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
