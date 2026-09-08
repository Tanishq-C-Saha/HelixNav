# HelixNav CP7 Codebase Audit
Date: 2026-08-20

Audit root: `source/helix_nav/helix_nav/tasks/manager_based/navigation/` (all file paths below are relative to this root). Every `.py` file under this tree was read in full, including `models/__init__.py` (confirmed empty stub — nothing to audit there) and every `__init__.py`.

## Framing note (read this before the findings below)

`config/go2/env_configs/cp7/cp7_env.py`, `cp7/__init__.py`, and `cp7/manager_configs/__init__.py` are each a single-line `# TODO(CP7): ...` comment. There is no `CP7` environment class, no scene/actions/observations/rewards/terminations/events defined anywhere under `cp7/`. A repo-wide `grep -rn "gym.register"` under this tree returns **zero matches** — no environment in this package is registered with gymnasium under any ID. `models/__init__.py` is empty.

The only functioning `ManagerBasedRLEnvCfg` in the whole tree is `HelixNavDebugBaseRLEnvCfg` (`config/go2/env_configs/debug/helixnav_debug_base_rl_env_cfg.py`), wired to `config/go2/env_configs/debug/manager_configs/*.py`. This is what the rest of this audit treats as "the live code" — it is a much earlier prototype than what the CP7 design document specifies (single reward term instead of 4, no CNN/path/lookahead observation, `num_envs` hardcoded to 1), and it has no train/play entry point: the only two scripts that instantiate it (`debug/manager_env.py`, `debug/rl_env_keyboard_controller.py`) are manual/keyboard control loops, not PPO training scripts. **As of this snapshot, CP7 as specified in the design document does not exist in code, and there is no way to launch RL training against any environment in this tree.** This is listed as Critical finding C1 below, and every subsequent finding about "the live env" refers to `HelixNavDebugBaseRLEnvCfg` and its wired manager configs, since that is the only candidate.

---

## Critical (will cause training failure or silent corruption)

### C1 — `config/go2/env_configs/cp7/cp7_env.py:1`, `cp7/__init__.py:1`, `cp7/manager_configs/__init__.py:1`, and no `gym.register` anywhere in the tree
- **Symptom**: There is no way to `gym.make()` or otherwise launch a CP7 environment. Any train script targeting a "CP7" task ID will fail with an unregistered-environment error before touching the sim.
- **Root cause**: `cp7_env.py` is literally `# TODO(CP7): Implement the CP7 RL-ready environment configuration.` and nothing else (0 lines of actual code). Same for the two adjacent `__init__.py` files. No file in the tree calls `gymnasium.register(...)`.
- **Fix**: Implement `cp7_env.py` (scene/actions/observations/rewards/terminations/events per the design doc), and add a `gym.register(id="Isaac-HelixNav-Go2-CP7-v0", entry_point=..., kwargs={"env_cfg_entry_point": ...})` call in `config/go2/env_configs/cp7/__init__.py`, following the pattern IsaacLab uses elsewhere (`config/go2/env_configs/__init__.py` should import and expose it too — it is currently empty).

### C2 — `mdp/events.py:168,171-174` — `_spawn_obstacles` indexes an undersized tensor with global env ids
- **Symptom**: `IndexError`/CUDA out-of-bounds crash the first time a *subset* of environments resets (i.e. almost immediately once `num_envs > 1` and episodes end at different times per env, which is the normal PPO training situation).
- **Root cause**:
  ```python
  def _spawn_obstacles(env, env_ids):
      env_origins = env.scene.env_origins
      for i, obs in enumerate(env._static_obstacles):
          pose = torch.zeros(len(env_ids), 7, device=env.device)     # shape (len(env_ids), 7)
          pose[env_ids, 0] = env._obstacles_pos[env_ids, i, 0] + env_origins[env_ids, 0]   # indexes with env_ids, not 0..len(env_ids)-1
          pose[env_ids, 1] = ...
          pose[env_ids, 2] = ...
          pose[env_ids, 3] = 1.0
          obs.write_root_com_pose_to_sim(pose, env_ids=env_ids)
  ```
  `pose` is allocated with exactly `len(env_ids)` rows, but is then written using `env_ids` (the *global* environment indices, e.g. `[2, 5, 9]`) as the row index. This only works by coincidence when `env_ids == arange(num_envs)` (a full reset of all envs from index 0). For any partial reset — the overwhelmingly common case in vectorized RL — `env_ids` contains values ≥ `len(env_ids)`, which is out of bounds for `pose`.
  The sibling function `_spawn_robot` two functions below (`mdp/events.py:179-216`) does this correctly — `root_state = asset.data.default_root_state[env_ids].clone()` then `root_state[:, 0] = ...` (positional slice, not `env_ids`-indexed) — confirming this is a copy/paste inconsistency, not an intentional pattern.
  This bug is currently masked because `helixnav_debug_base_rl_env_cfg.py:41` hardcodes `self.scene.num_envs = 1`, where `env_ids` is always `[0]` and the bug is unobservable.
- **Fix**:
  ```python
  pose = torch.zeros(len(env_ids), 7, device=env.device)
  pose[:, 0] = env._obstacles_pos[env_ids, i, 0] + env_origins[env_ids, 0]
  pose[:, 1] = env._obstacles_pos[env_ids, i, 1] + env_origins[env_ids, 1]
  pose[:, 2] = env._obstacles_pos[env_ids, i, 2] + env_origins[env_ids, 2]
  pose[:, 3] = 1.0
  obs.write_root_com_pose_to_sim(pose, env_ids=env_ids)
  ```

### C3 — No PBRS progress reward term is wired; the design's core dense-reward fix does not exist in the live reward config
- **Symptom**: Reward is exactly 0.0 for the entire episode except on steps where the robot is within the (buggy, see C4) goal radius. With a sparse-only signal and no path-progress shaping, PPO will see almost no reward gradient and is very likely to fail to converge — this reproduces the exact class of failure (Failure 3, "drifts to open space") the CP7 design doc says PBRS was built to fix, except here the fix was never implemented at all.
- **Root cause**: `mdp/events.py:112-116` computes and stores `env._path_remaining` / `env._prev_path_remaining` (arc length along the A* path) on every reset — this is the infrastructure for `Φ(s) = -path_remaining_distance`. But `grep -rn "_path_remaining"` across the whole tree shows these fields are **only ever written, never read** anywhere. `config/go2/env_configs/debug/manager_configs/rewards.py` (the only `RewardCfg` in the tree) contains exactly one term:
  ```python
  @configclass
  class RewardCfg:
      goal_reached = RewTerm(func=mdp.goal_reached_reward, params={"min_goal_threshold": 0.5}, weight=50)
  ```
  No `r_progress = γΦ(s') − Φ(s)` term, and no smoothness term (see H3) exists.
- **Fix**: Implement a `progress_pbrs` reward function that reads `env._path_remaining`/`env._prev_path_remaining` (updating `_prev_path_remaining = _path_remaining` each step after computing the new path-remaining distance), and add it as a `RewTerm` in `RewardCfg` per design doc §4.2.

### C4 — Goal-reach reward and goal-reach termination use different thresholds, creating an infinite reward-farming loophole
- **Symptom**: Robot can hover in an annulus around the goal and accumulate the +50-weighted reward every physics step indefinitely without the episode ever terminating — a reward-hacking exploit of exactly the kind the design doc calls out as the CP6 "Failure 2" pattern (§1.1, §4.6), just via a different mechanism (band-camping instead of standing still).
- **Root cause**: `config/go2/env_configs/debug/manager_configs/rewards.py:13` sets `min_goal_threshold: 0.5`. `config/go2/env_configs/debug/manager_configs/terminations.py:26` sets `min_distance_threshold: 0.3`. `mdp/rewards.py:14` even has a comment on the function's own default acknowledging this: `min_goal_threshold: float = 0.5  #! should be same as terminations goal reached dist` — but the wired termination value (0.3) does not match either the reward's wired value (0.5) or the reward function's own documented-matching default (0.5).
  Compounding this: `mdp/rewards.py:26` — `return (dist <= min_goal_threshold).float()` — is **not** edge-triggered; it returns `1.0` on every step the condition holds, not once. So for any step the robot sits between 0.3 m and 0.5 m from goal, it collects the reward and the episode does not end.
- **Fix**: Use the same threshold value for both (e.g. import a single `GOAL_RADIUS` constant used by both `RewardCfg.goal_reached` and `TerminationsCfg.goal_reached`), so the reward-granting step and the terminating step are the same step by construction.

---

## High (likely to degrade training or cause intermittent bugs)

### H1 — `config/go2/env_configs/debug/manager_configs/scene.py:133` — collision sensor regex alternation bug
```python
collision_sensor = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/robot/.*_(hip|thigh)|Head_(upper|lower)",
    update_period=0.0,
)
```
- **Symptom**: Collision detection silently only covers hip/thigh bodies; head-body contacts are not scoped under the robot's env-namespaced path at all.
- **Root cause**: Regex `|` has the lowest precedence, so this string parses as two independent alternatives: `{ENV_REGEX_NS}/robot/.*_(hip|thigh)` OR `Head_(upper|lower)`. The second alternative has no `{ENV_REGEX_NS}/robot/` prefix, so it can never match a real per-env body path — it would only match a literal top-level prim named exactly `Head_upper`/`Head_lower`, which does not exist under the robot's namespaced path. This is the sensor that feeds `mdp.terminations_by_collisions` (`mdp/terminations.py:13-28`), i.e. it directly drives the collision termination.
- **Fix**: Group the whole alternation under the shared prefix: `prim_path="{ENV_REGEX_NS}/robot/.*(_hip|_thigh|Head_upper|Head_lower).*"` (exact suffix needs verifying against real Go2 body names in the USD, but the prefix must wrap all alternatives).

### H2 — `map_generators/random_map_generator.py:306-316` — `generate_with_retry` crashes training instead of falling back
```python
def generate_with_retry(self, difficulty: int, seed: int, max_attempts: int = 20) -> MapSpec:
    for attempt in range(max_attempts):
        result = self.generate(difficulty, seed + attempt)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate difficulty={difficulty} map after {max_attempts} attempts...")
```
- **Symptom**: If 20 consecutive seeds fail to produce a valid map (plausible at difficulty 3, `count=(12,13)` obstacles with `min_spacing=0.4` in a 12×12 arena with `START_GOAL_OBS_MARGIN`/`START_GOAL_WALL_MARGIN` constraints), the reset event raises an unhandled `RuntimeError`, which propagates out of `EventManager.apply(mode="reset")` and kills the whole training run (not just the unlucky env).
- **Root cause**: The design doc (§5.3, step 6) explicitly requires: "If validation fails, resample and try again. Cap retries at 10; if all fail, use a fallback simple configuration." No fallback exists — only the exception.
- **Fix**: Catch the `RuntimeError` in `mdp/events.py:reset_map_and_spawn` (or inside `generate_with_retry` itself) and fall back to a known-good fixed `MapSpec` (e.g. the difficulty-1 config with a fixed seed) rather than propagating.

### H3 — Smoothness regularizer reward term never implemented
- **Symptom**: Nothing in the design doc's §4.5 (`r_smooth = -0.01 * ||a_t - a_{t-1}||^2`) is present. Combined with `mdp/events.py:251` (`env._prev_actions` initialized) and `mdp/events.py:159` (`env._prev_actions[env_ids] = 0.0` on reset), this is the same dead-groundwork pattern as C3 — state is tracked but never read by any reward function (confirmed via `grep -rn "_prev_actions"`, only writes appear). Without this term, PPO is free to learn bang-bang command profiles per the design doc's own stated risk (§4.5).
- **Fix**: Implement `action_smoothness` reward reading `env.action_manager.action` vs `env._prev_actions`, add to `RewardCfg`, and update `env._prev_actions` every step (not just on reset).

### H4 — `mdp/pre_trained_policy_action.py:87-88` — no clipping on the raw nav-policy action before it becomes the locomotion policy's velocity command
```python
def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
```
- **Symptom**: Whatever the high-level nav policy outputs is passed straight through as `[vx, vy, yaw_rate]` to the frozen locomotion policy's `velocity_commands` observation (`__init__` wires `cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions` at line 59) with no `torch.clamp`.
- **Root cause**: The design doc explicitly attributes the CP6.5 NaN-loss failure to "unbounded reward terms multiplied by velocities that could reach 8 to 15 m/s during physics instability" (§6.2) and lists explicit runtime guards CP7 is supposed to add. No clamp exists at this specific boundary — the one place an unbounded PPO Gaussian action sample would otherwise be caught before reaching the physics-critical low-level policy.
- **Fix**: `self._raw_actions[:] = torch.clamp(actions, min=-max_cmd, max=max_cmd)` with `max_cmd` matching whatever range the nav policy's action distribution/network head is designed to produce (not verifiable further — `models/` is an empty stub, see framing note).

### H5 — Live observation space is a small fraction of the CP7 design spec; the stated root cause of Failure 1 is not addressed
- **Symptom**: The design doc identifies "insufficient observation context for obstacle bypass" as the root cause of CP6's obstacle-circling failure and specifies an 80×80×3 CNN input (occupancy, rendered A* path, goal indicator) plus 44 MLP scalars including 8 lookahead points (24 dims) + 8 goal flags + explicit relative goal + proprioception (§3). The live `ObservationsCfg` (`config/go2/env_configs/debug/manager_configs/observations.py:12-63`) has exactly: `prev_actions` (duplicated, see M4), `current_lin_vel`, `current_ang_vel`, `rel_goal_w` (3 dims), and raw `depth_images` (not the processed occupancy/path/goal 3-channel grid). There is no rendered A* path channel, no goal-indicator channel, no lookahead vectors, no lookahead goal flags, and no `projected_gravity` term (present in CP6 per the doc's own component table, absent here).
- **Root cause**: This observation set predates the CP7 redesign; it was never migrated (consistent with C1 — the "CP7" package that should contain this work is an empty stub).
- **Fix**: Not a one-line fix — implement the observation redesign in §3 of the design doc (this is essentially CP7.2 from the doc's own implementation roadmap, §7.2), against a real `cp7_env.py`.

### H6 — No curriculum manager; `env._current_difficulty` never advances during actual RL training
- **Symptom**: `mdp/events.py:230` sets `env._current_difficulty = 1` once at init and nothing in `HelixNavDebugBaseRLEnvCfg` (`config/go2/env_configs/debug/helixnav_debug_base_rl_env_cfg.py:21-44`) ever changes it — there is no `curriculum` field on the config at all. The only place difficulty is incremented is `config/go2/env_configs/debug/manager_env.py:46-48`, a manual debug script using `ManagerBasedEnv` (not the RL config) that bumps difficulty every 100 steps on a fixed schedule unrelated to success rate.
- **Root cause**: Design doc §5.2 requires curriculum advancement "triggered by rolling-average success rate crossing thresholds" specifically because "full randomisation from step 0 is unstable." As currently wired, any real training run against `HelixNavDebugBaseRLEnvCfg` would either train exclusively at difficulty 1 forever, or (if someone wires the debug script's naive step-count bump into training) advance without regard to whether the policy is actually succeeding — both are exactly the failure mode the doc warns against.
- **Fix**: Add a `CurriculumTermCfg` wired to rolling success rate (as tracked via the termination/reward extras) per design doc §5.2.

---

## Medium (correctness risk under edge cases)

### M1 — `mdp/observations.py:79` — `norm_dist` can exceed 1.0, unbounded/unclipped network input
```python
norm_dist = dist/arena_size   # arena_size passed as 12.0 (manager_configs/observations.py:47)
```
- The 12×12 m arena's diagonal is ≈16.97 m, so a robot/goal placed near opposite corners yields `norm_dist ≈ 1.41`, not in `[0,1]` as intended for a normalized network input. Not clamped anywhere downstream.
- **Fix**: `norm_dist = torch.clamp(dist / arena_size, max=1.0)` or normalize by the true max possible distance (arena diagonal) instead of `arena_size`.

### M2 — Goal bonus weight is not compensated for `RewardManager`'s automatic `dt` scaling
- IsaacLab's `RewardManager` multiplies each term's `weight` by `env.step_dt` before summing (confirmed in `isaaclab/managers/reward_manager.py` docstring: "The reward manager multiplies the reward term's `weight` with the time-step interval `dt`"). `config/go2/env_configs/debug/manager_configs/rewards.py:15` sets `weight=50` directly. With `sim.dt=1/200` and `decimation=20` (`helixnav_debug_base_rl_env_cfg.py:38-39`), `step_dt=0.1s`, so the actually-delivered bonus is `50 * 0.1 = 5.0`, not the `+50` the design doc calibrates against in §4.3 ("approximately ten times the cumulative PBRS shaping reward... over a typical successful episode"). Independent of the C4 threshold-mismatch bug, this is a 10x calibration miss relative to the design doc if/when this term is otherwise fixed.
- **Fix**: Either set `weight=500` to compensate for the `dt` multiplication, or make the goal term a one-shot un-scaled bonus applied outside the standard weighted-sum path.

### M3 — Implemented difficulty/curriculum buckets don't match the design doc's staged spec
`map_generators/random_map_generator.py:56-60`:
```python
DIFFICULTY_CONFIGS = {
    1: {"count": 5, ...},
    2: {"count": (8, 10), ...},
    3: {"count": (12, 13), ...},
}
```
- Design doc §5.2 specifies Stage 1 as "fixed simple map, **single obstacle**, straight-line optimal path," but difficulty 1 here places 5 obstacles. Doc's obstacle-count set is `{5, 8, 12, 15}` (§5.1); implementation uses `5`, `(8,10)`, `(12,13)` — a reasonable but undocumented deviation. Combined with H6 (no curriculum), this is currently moot for training but will need reconciling once a curriculum manager is added.

### M4 — `config/go2/env_configs/debug/manager_configs/observations.py:20-22` and `:38-40` — duplicate `prev_actions` attribute
```python
prev_actions = ObsTerm(func=mdp.last_action)      # line 20-22
...
current_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # line 29-31
current_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # line 33-35
prev_actions = ObsTerm(mdp.last_action)           # line 38-40, silently overwrites the first
```
- Two class-body assignments to the same attribute name; the second (positional-arg call, no `func=` keyword) silently wins per normal Python class-body semantics. Both reference the same function so there's no behavioral difference today, but it reads as if two distinct observations were intended and one is a landmine for a future edit that changes only one of the two copies.
- **Fix**: Delete the duplicate at lines 38-40.

### M5 — `mdp/events.py:100-101` vs `:115-116` — inconsistent partial reset of path state when A* fails
```python
env._path_lengths[env_id] = 0
env._paths_world[env_id] = 0.0

if path is not None:
    ...
    env._path_remaining[env_id] = arc_length
    env._prev_path_remaining[env_id] = arc_length
```
- If `astar()` returns `None` (path not found — see the docstring in `map_generators/astar.py:1-7` claiming this "shouldn't" happen given matching BFS/A* connectivity, but the code doesn't structurally guarantee it, e.g. `inflate_grid` in `events.py:96` uses `cells=1` while the map generator's own BFS validation inflation is a separately-maintained constant `BFS_INFLATION_CELLS=1` in `random_map_generator.py:69` — currently equal, but nothing enforces they stay equal), `env._path_lengths`/`env._paths_world` are zeroed for the new episode, but `env._path_remaining`/`env._prev_path_remaining` are **not** touched and retain the *previous episode's* value. This is currently harmless only because nothing reads `_path_remaining` yet (C3) — the moment PBRS is wired up per the recommended fix for C3, this becomes exactly the "stale-path bug" the design doc calls "silent and catastrophic" in §5.3.
- **Fix**: Move `env._path_remaining[env_id] = 0.0` / `env._prev_path_remaining[env_id] = 0.0` above the `if path is not None:` check, alongside the other unconditional resets.

---

## Low (code quality, performance, style)

- **`config/go2/env_configs/debug/keyboard_controller_debugger.py:110`** and **`config/go2/env_configs/debug/rl_env_keyboard_controller.py:110`** — both reference an undefined variable `depth_images` inside the `if args.save_camera_images:` branch (`images=[rgb_images[i], depth_images[i]]`); would raise `NameError` if that flag is ever used. Debug-only scripts, not part of any training path — **dead code, not wired into CP7**.
- **`mdp/utils.py:24-25`** — leftover `print(f"[DEBUG]: child: ...")` / `print(f"[DEBUG]: child_name: ...")` inside `get_static_obstacles`, which runs once at `init_nav_state` time (not a hot per-step path, but noisy/unconditional debug prints left in).
- **`mdp/utils.py:62`** — leftover `print(f"[DEBUG]: Obstacle pool = ...")`, same category.
- **`mdp/events.py:33`** — `GRID_CELLS = 61` is a locally re-declared magic constant duplicating `grid_utils.CELLS` (also 61), which this same file already imports (`from ...map_generators import (..., CELLS, ...)` at line 14) but doesn't use for this purpose. Drift risk if grid resolution ever changes.
- **`config/go2/env_configs/debug/manager_configs/events.py:11-17`** — `reset_map_and_spawn` is wired with `params={"visualize_map": True}` unconditionally, meaning every single reset calls `plot_nav_state` and (if the raycaster path is hit) `visualize_nav_states`, each doing full matplotlib figure creation + PNG save to disk. Harmless at the current hardcoded `num_envs=1` (`helixnav_debug_base_rl_env_cfg.py:41`), but would become a severe per-reset I/O/CPU bottleneck the instant `num_envs` is raised for real PPO training, with no config-level toggle to disable it short of editing source.
- **`config/go2/env_configs/cp7/`** (all three files) — **dead code, not wired into CP7** in the sense that it isn't wired into anything at all; already covered under Critical C1, listed here only to satisfy the "explicitly note dead code" requirement.
- **`map_generators/random_map_generator.py:29-50`** — module-level default `OBSTACLE_POOL` (20 entries) is never actually used by the live env: `mdp/events.py:227` always constructs `RandomMapGenerator(obstacle_pool=obstacle_pool)` with the real, dynamically-scanned 13-entry pool from the live scene (`mdp/utils.py:get_obstacle_pool`), overriding the class default. The 20-entry list is only exercised by the standalone `visualize_maps.py`/`visualize_single_map.py` demo scripts (which construct `RandomMapGenerator()` with no args) — **dead code for the live env**, real code for the demo scripts. Worth knowing so nobody "fixes" an apparent 20-vs-13 obstacle-pool-size mismatch that doesn't actually affect training.
- **`config/go2/env_configs/debug/helixnav_debug_base_env_cfg.py:32`** / **`helixnav_debug_base_rl_env_cfg.py:40`** — `env_spacing = 12.2` against an arena whose walls span roughly `[-6.05, 6.05]` (wall thickness 0.1 m centered at ±6.0, per `scene.py:141-158` etc.) leaves only ~0.1 m clearance between adjacent environments' outer wall faces. Walls are solid/kinematic so this isn't currently a physics-bleed bug, but it's a thin margin with no headroom for any future wall-size tweak.

---

## Verified Correct (checked and confirmed NOT bugs — so no one re-checks these)

- **`map_generators/grid_utils.py:22-36`** — `world_to_grid`/`grid_to_world` use a consistent `grid[row, col]` with `row=Y, col=X` convention in both directions; cross-checked against `map_generators/astar.py` and `mdp/events.py:105` (`grid_to_world(r, c)` called with the same `(row, col)` ordering A* produces).
- **`map_generators/astar.py`** vs **`map_generators/bfs.py`** — both use the identical 8-connected neighbor offset list; `astar.py`'s own docstring (lines 1-7) states this is intentional so that any map `bfs_reachable()` accepts is guaranteed to have an A* path, and the neighbor/cost tables in both files match exactly.
- **`map_generators/events.py:96` inflation (`cells=1`)** vs **`map_generators/random_map_generator.py:69` `BFS_INFLATION_CELLS=1`** — both currently use the same 1-cell (0.2 m) inflation amount for A* planning and BFS reachability validation respectively, so a map that passes generation-time validation is not spuriously blocked or falsely accepted at A*-planning time.
- **`mdp/observations.py:51-81`** (`get_relative_goal_vector`) — correctly transforms the world-frame goal vector into the robot's yaw frame via `math_utils.quat_apply_inverse(robot_yaw_quat, ...)` using IsaacLab's wxyz quaternion convention, and guards the unit-direction division with `dist.clamp(min=1e-6)` (line 75) to avoid a NaN when the robot is exactly at the goal.
- **`mdp/terminations.py:32-49`** (`goal_reached`) and **`:13-28`** (`terminations_by_collisions`) — both avoid corrupting persistent buffers: `goal_reached` operates on `env._goal_positions + env.scene.env_origins` (a freshly-allocated tensor from the `+`) and `robot.data.root_pos_w.clone()` before in-place z-zeroing.
- **`mdp/events.py:179-216`** (`_spawn_robot`) — correctly indexes its reset-sized `root_state` tensor positionally (`root_state[:, 0] = ...`), unlike the sibling bug in `_spawn_obstacles` (see C2).
- **`config/go2/env_configs/debug/manager_configs/terminations.py:32-35`** — `time_out = DoneTerm(func=mdp.time_out, time_out=True)` resolves to IsaacLab core's standard `time_out` function via the wildcard import in `mdp/__init__.py:1` (`from isaaclab.envs.mdp import *`), correctly routing episode timeouts to the truncation/`time_outs` buffer rather than `terminated`, matching the manager-based RL env convention.
- **Collision handling is termination-only** (`config/go2/env_configs/debug/manager_configs/terminations.py:14-20`, `RewardCfg` has no collision-penalty term) — matches design doc §4.4 exactly; no accumulating collision penalty exists that could incentivize the "suicide problem" the doc warns about.
- **`config/go2/env_configs/debug/manager_configs/scene.py:59-66`** (`height_scanner`, `pos=(0,0,20.0)`) — this large-looking Z offset is the canonical IsaacLab height-scanner pattern (mount far above the robot, raycast straight down at the ground so ray-hit distance minus offset gives terrain height, avoiding self-occlusion by the robot body); not a bug.
- **`config/go2/env_configs/debug/manager_configs/scene.py:83-84` and `:111-112`** (`rgb_camera`/`depth_camera` offset `rot=(0.5,-0.5,0.5,-0.5)`, `convention="ros"`) — this is IsaacLab's standard forward-facing-camera quaternion convention, applied consistently to both cameras.
- **`mdp/observations.py:21-46`** (`get_depth_images`) — correctly replaces `inf` (missed raycasts, from `depth_clipping_behavior="none"`) with `max_distance` before clamping and normalizing, preventing `inf`/`NaN` from reaching the observation buffer.
- **Reset ordering vs. sensor staleness**: confirmed against IsaacLab core (`isaaclab/envs/manager_based_env.py:556-568`) that `_reset_idx` calls `self.scene.reset(env_ids)` (which flags every sensor `_is_outdated=True`) **before** `self.event_manager.apply(mode="reset", ...)` (where `reset_map_and_spawn` repositions obstacles/robot). So the next `.data` read on `depth_camera`/`occupancy_scanner` after a reset is guaranteed to trigger a fresh raycast against the *new* obstacle layout, not stale previous-episode data — the exact class of bug the reference doc's §4 warns about does not manifest here.
- **`config/go2/env_configs/debug/manager_configs/scene.py`** obstacle/wall `RigidObjectCfg` entries (`s_obs_0`–`s_obs_12`, `wall_north/south/east/west`) — all set `disable_gravity=True, kinematic_enabled=True` plus explicit `CollisionPropertiesCfg()` on mesh-based `CuboidCfg`/`CylinderCfg`/`ConeCfg` spawners (not pure-visual primitives) — valid raycast targets, won't fall over or get physically pushed.
- **`models/__init__.py`** — confirmed empty; no network/policy code exists in this checkpoint to audit for the "models" category.
