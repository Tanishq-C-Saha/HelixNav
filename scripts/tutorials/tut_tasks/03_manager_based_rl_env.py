"""
Goal:

Config-only exercise (mostly) — build RewardsCfg, TerminationsCfg,
wire them into a CartpoleEnvCfg(ManagerBasedRLEnvCfg), run with
random actions, watch it reset itself when it fails, read the info dict.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

# adding custom args :
parser.add_argument(
    "--num_envs", help="No. of parallel envs to be made.", default=4, type=int
)

AppLauncher.add_app_launcher_args(parser=parser)

cli_args = parser.parse_args()

print(cli_args)

# setting up the omniverse context
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app


"""Everything rest follows."""

from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.envs import mdp

from isaaclab.scene import InteractiveSceneCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils

import torch
import math


# custom scene
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """My custom scene.
    1. ground
    2. cartpole cfg
    3. light
    """

    # ground
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    # robot articulation
    cartpole = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/cartpole")

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.6, 0.6)),
    )


# actions
@configclass
class ActionCfg:
    """Action terms for the cartpole."""

    slider_velocity = mdp.JointVelocityActionCfg(
        asset_name="cartpole", joint_names=["slider_to_cart"], scale=5.0
    )


#  observations
@configclass
class ObservationsCfg:
    """Observations for the Cartpole."""

    # policy cfg
    @configclass
    class PolicyCfg(ObsGroup):
        """Custom Policy Observations."""

        # how do we determine what the policy shopuld see
        """
        in our case : 
        1. maybe last action
        2. maybe current velocity 
        3. maybe current pole angle 
        4. maybe current pose (env specific)
        """

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="cartpole")},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="cartpole")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # policy
    policy: ObsGroup = PolicyCfg()


# events
@configclass
class EventsCfg:
    """Let there be a simple event that resets the env at random positions.
    So that the system is more robust and policy learns to balance
    the cartpole from diff states & not just one!

    # 1. On satrtup : randomize masses of the poles
    # 2. On reset :
        a. randomize pole pos and velocity
        b. randomize cart position and velocity
    """

    # satrtup
    randomize_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cartpole"),
            # 0.1kg to 3kg can be added to default mass
            "mass_distribution_params": (0.1, 3),
            "operation": "add",
        },
    )

    # reset
    resetcart_cart_pos = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cartpole", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.3, 0.3),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cartpole", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


# helper function for join position reward
def joint_pos_target_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target: float
):
    """Penalize for deviating from target : Sum of Squared error"""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids].clone())

    return torch.sum(torch.square(joint_pos - target), dim=-1)


# rewards
@configclass
class RewardsCfg:
    """
    1. Alive Reward: Encourage the agent to stay alive for as long as possible.
    2. Terminating Reward: Similarly penalize the agent for terminating.
    3. Pole Angle Reward: Encourage the agent to keep the pole at the desired upright position.
    4. Cart Velocity Reward: Encourage the agent to keep the cart velocity as small as possible.
    5. Pole Velocity Reward: Encourage the agent to keep the pole velocity as small as possible.
    """

    # alive reward
    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # termination penalty
    termination = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )

    # Pole angle deviation penalty
    pole_pos = RewTerm(
        func=joint_pos_target_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(name="cartpole", joint_names=["cart_to_pole"]),
            "target": 0.0,
        },
    )

    # Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("cartpole", joint_names=["slider_to_cart"])
        },
    )

    # Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("cartpole", joint_names=["cart_to_pole"])},
    )


# terminations
@configclass
class TerminationsCfg:
    """Terminations for the cartpole env.
    1. episode time out
    2. pole below certain pos
    """

    # episode time out
    episode_time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # pole below certain pos
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("cartpole", joint_names=["slider_to_cart"]),
            "bounds": (-3.0, 3.0),
        },
    )


# commands
"""No commands for now"""


@configclass
class MyRlEnvCfg(ManagerBasedRLEnvCfg):
    """
    Design choices :
    1. ManagerBasedRl env
        a. Scene
        b. Action
        c. Obs
        d. events
        ------ RL Env specific ------
        e. Rewards
        f. terminations
        g. curriculum
        h. commands

    """

    # scene
    scene: InteractiveSceneCfg = MySceneCfg()

    # actions
    actions = ActionCfg()

    # observations
    observations = ObservationsCfg()

    # events
    events = EventsCfg()

    # rewards
    rewards = RewardsCfg()

    # terminations
    terminations = TerminationsCfg()

    # commands

    # curriculumn

    """Note : commands and curriculum not implemented."""

    def __post_init__(self):
        """Post intialization for changing env related settings."""

        self.sim.dt = 1.0 / 200.0  # simulation running @ 200 Hz ~ 0.005ms
        self.decimation = 4  # env running @ 50 Hz ~ 0.02ms

        # rendering of viewport graphics @ 50 hz ~ 50FPS
        self.sim.render_interval = self.decimation

        # to provide fresh PhsyX data at env reset
        self.num_rerenders_on_reset = 1

        self.episode_length_s = 30  # 30s

        self.scene.num_envs = cli_args.num_envs  # setting 4 by default
        self.scene.env_spacing = 5  # 5m of env spacing


# main
def main():
    """Main func implementation."""
    cartpole_rl_env_cfg = MyRlEnvCfg()
    cartpole_env = ManagerBasedRLEnv(cfg=cartpole_rl_env_cfg)

    cartpole_env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():

            # random action
            joint_action = torch.rand_like(cartpole_env.action_manager.action)
            obs, rew, terminated, time_outs, extras = cartpole_env.step(
                action=joint_action
            )

            term_names = cartpole_env.reward_manager._term_names.copy()
            term_rewards = cartpole_env.reward_manager._step_reward.clone()

            # debug
            print(f"\n\n[DEBUG]: obs = {obs}\n")
            print(f"[DEBUG]: rew = {rew}\n")
            print(f"[DEBUG]: terminations = {terminated}\n")
            print(f"[DEBUG]: time out = {time_outs}\n")
            print(f"[DEBUG]: extras = {extras}\n")

            print("\n[Sp. DEBUG]: per step reward \n")
            for i, term_name in enumerate(term_names):
                print(f"{term_name} : {term_rewards[:, i]}")


    cartpole_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()


"""
Some imp conclusions : 
1. Obs come as dictionary of ObsGroup and tensors 
    obs = {"policy" : tensor}
2. rewards = tensor() of dim [N_evs] *NOTE* cumulative reward for each env
3. terminations and time out as boolean tensor 
4. extras consist of imp specific details : 


sample info:
[DEBUG]: obs = {'policy': tensor([[ 1.4918,  2.7522,  0.4659,  7.9022],
        [ 2.5828,  3.7314,  3.3529, -5.4281],
        [ 0.0568,  2.0363,  0.8043,  7.0973],
        [ 2.7237,  3.4402,  3.8005, -6.4838]], device='cuda:0')}

[DEBUG]: rew = tensor([-0.3019, -0.2597, -0.1647, -0.3227], device='cuda:0')

[DEBUG]: terminations = tensor([False, False, False, False], device='cuda:0')

[DEBUG]: time out = tensor([False, False, False, False], device='cuda:0')

[DEBUG]: extras = {'log': {
         'Episode_Reward/alive': tensor(0.0066, device='cuda:0'),
         'Episode_Reward/termination': tensor(-0.0033, device='cuda:0'),
         'Episode_Reward/pole_pos': tensor(-0.4891, device='cuda:0'),
         'Episode_Reward/cart_vel': tensor(-0.0011, device='cuda:0'),
         'Episode_Reward/pole_vel': tensor(-0.0013, device='cuda:0'),
         'Episode_Termination/episode_time_out': 0.0,
         'Episode_Termination/cart_out_of_bounds': 1.0}}

"""


# special debug info 
"""
[DEBUG]: obs = {'policy': tensor([[ 1.0079,  3.7964,  1.2263,  6.2514],
        [ 1.3035,  3.1758, -0.2718,  7.8893],
        [ 0.9017,  5.0180,  2.3109, -0.7754],
        [ 0.2343,  3.9103,  1.6717,  4.2684]], device='cuda:0')}

[DEBUG]: rew = tensor([-0.2462, -0.3851, -0.0626, -0.2240], device='cuda:0')

[DEBUG]: terminations = tensor([False, False, False, False], device='cuda:0')

[DEBUG]: time out = tensor([False, False, False, False], device='cuda:0')

[DEBUG]: extras = {'log': {'Episode_Reward/alive': tensor(0.0061, device='cuda:0'), 'Episode_Reward/termination': tensor(-0.0033, device='cuda:0'), 'Episode_Reward/pole_pos': tensor(-0.5528, device='cuda:0'), 'Episode_Reward/cart_vel': tensor(-0.0011, device='cuda:0'), 'Episode_Reward/pole_vel': tensor(-0.0011, device='cuda:0'), 'Episode_Termination/episode_time_out': 0.0, 'Episode_Termination/cart_out_of_bounds': 1.0}}


[Sp. DEBUG]: per step reward 

alive : tensor([0.1000, 0.1000, 0.1000, 0.1000], device='cuda:0')
termination : tensor([-0., -0., -0., -0.], device='cuda:0')
pole_pos : tensor([-12.3682, -19.3118,  -3.2015, -11.2608], device='cuda:0')
cart_vel : tensor([-0.0123, -0.0027, -0.0231, -0.0167], device='cuda:0')
pole_vel : tensor([-0.0313, -0.0394, -0.0039, -0.0213], device='cuda:0')


"""


#! note
"""
Some imp conclusions : 
1. Obs come as dictionary of ObsGroup and tensors 
    obs = {"policy" : tensor}
2. rewards = tensor() of dim [N_evs] *NOTE* cumulative reward for each env
3. terminations and time out as boolean tensor 
4. extras consist of imp specific details : 
    - extras["log"] is NOT refreshed every step. It's only overwritten inside
      _reset_idx(), i.e. only on steps where at least one env actually resets.
      On steps where reset_env_ids is empty, extras["log"] just holds whatever
      was last written -> it will look "stale" relative to the live rew/obs.
    - 'Episode_Reward/<term>' values are per-episode summaries (mean contribution
      over the episode that just ended), logged only for the env(s) that reset
      this call -- NOT a live per-step, per-env breakdown.
    - 'Episode_Termination/<term>' values (e.g. 0.0 / 1.0) are the FRACTION of
      resetting envs whose episode ended due to that specific termination term.
      i.e. "of the envs that reset just now, what % ended because of me."
      Useful later for diagnosing HelixNav failures: is the policy actually
      failing (collision-type term firing), or just running out of episode
      budget (time-out term firing)?

5. For LIVE, per-step, per-env, per-term reward values -- don't use extras["log"].
   RewardManager already keeps exactly this in an internal buffer:
       reward_manager._step_reward   -> shape [num_envs, num_terms]
       reward_manager._term_names    -> ordered list of term names (columns line up)
   No source edits, no subclassing RewardManager needed -- just read the attribute.

6. UNIT MISMATCH TRAP (confirmed empirically, don't skip this):
   Inside RewardManager.compute():
       value = term_cfg.func(...) * term_cfg.weight * dt      # dt-scaled
       self._reward_buf += value                               # <- what `rew` returns
       self._step_reward[:, term_idx] = value / dt              # <- dt DIVIDED BACK OUT
   So `_step_reward` is a dt-NORMALIZED "reward rate", not the actual per-step
   contribution that gets summed into `rew`. Summing raw _step_reward columns
   will NOT equal rew (was off by ~50x here, since step_dt = sim.dt*decimation
   = (1/200)*4 = 0.02, i.e. 1/0.02 = 50).
   Fix: multiply back by step_dt before comparing/interpreting against rew:
       (step_reward[:, i] * cartpole_env.step_dt)
   Always sanity-check a debug tool's units against a known-good reference
   (here: does my per-term sum equal the real `rew`?) before trusting it to
   guide real reward-weight tuning -- e.g. don't conclude "pole_pos dominates
   1000x" from raw _step_reward without rescaling first.
"""

