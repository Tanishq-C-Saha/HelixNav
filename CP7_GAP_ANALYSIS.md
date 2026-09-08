# CP7 Environment Gap Analysis

Generated: 2026-09-03
Audit scope: read in full, end-to-end (no skimming), 21 Python files, 2290 lines total.

**`mdp/` (948 lines)**
- `mdp/commands.py` (108)
- `mdp/events.py` (263)
- `mdp/__init__.py` (10)
- `mdp/observations.py` (86)
- `mdp/pre_trained_policy_action.py` (185)
- `mdp/rewards.py` (26)
- `mdp/terminations.py` (49)
- `mdp/utils.py` (69)
- `mdp/visualize_utils.py` (152)

**`config/go2/env_configs/debug/` (1342 lines)**
- `debug_viz_utils.py` (113)
- `helixnav_debug_base_env_cfg.py` (37)
- `helixnav_debug_base_rl_env_cfg.py` (47)
- `__init__.py` (0, empty)
- `keyboard_controller_debugger.py` (138)
- `manager_configs/actions.py` (39)
- `manager_configs/events.py` (17)
- `manager_configs/__init__.py` (6)
- `manager_configs/observations.py` (63)
- `manager_configs/rewards.py` (15)
- `manager_configs/scene.py` (607)
- `manager_configs/terminations.py` (35)
- `manager_env.py` (79)
- `rl_env_keyboard_controller.py` (146)

(`manager_configs/__pycache__/*.pyc` excluded — build artifacts, not source.)

## Summary

- Total spec items checked: 46
- Present and correct: 19
- Present but incorrect: 17
- Missing entirely: 10
- Critical bugs found: 9

---

## Section 1: NavigationWaypointCommand

**Status: PARTIAL** — the class exists and is structurally shaped like a `CommandTerm`, but it is **not wired into any env config**, two of its three core computation methods are unimplemented stubs, and it references an attribute that does not exist anywhere in the codebase.

Findings:
- **CommandTerm subclass exists**: PRESENT — `commands.py:13`, `class NavigationWaypointCommand(CommandTerm)`.
- **`goal_positions` tensor**: MISSING — no such tensor is created on `self` anywhere in `commands.py:19-48` (`__init__`). Goal state instead lives on the env itself as `env._goal_positions`, created in `events.py:246` (`init_nav_state`) and populated in `events.py:86-88`.
- **`paths_world` tensor**: INCORRECT — a similarly-named tensor exists (`self.path_world`, `commands.py:33`, note singular "path" not "paths"), but it is populated from `self._env._paths_local` (`commands.py:58`), an attribute that **does not exist anywhere in the repository** (verified by grep — only reference is this one line). This is a broken attribute reference; `_resample` would raise `AttributeError` the moment it runs.
- **`path_remaining` tensor**: INCORRECT — tensor declared (`commands.py:42`), but the only function that would populate it, `_compute_path_remaining` (`commands.py:101-104`), is an unimplemented stub (`# TODO ... pass`). It never advances past its `__init__`-time value of 0.
- **`prev_path_remaining` tensor**: INCORRECT — same issue; `commands.py:43`, `88`, `66` all reference it, but since `path_remaining` is never actually computed, `prev_path_remaining` only ever tracks a value that is permanently 0.
- **`lookaheads` tensor, 8 slots at [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]m**: INCORRECT — `SAMPLE_DISTANCES` (`commands.py:16`) and the tensor shape `(N, 8, 3)` (`commands.py:38`) are correct, but `_compute_lookaheads` (`commands.py:92-99`) is an unimplemented stub (`# TODO ... pass`). Lookaheads are never computed.
- **`occupancy_grids` tensor (61×61 @ 0.2m, 12×12m arena)**: MISSING from the `CommandTerm` — no such tensor is created in `commands.py`. The equivalent state lives on the env as `env._occupancy_grids`, sized `(N, GRID_CELLS, GRID_CELLS)` with `GRID_CELLS = 61` (`events.py:33, 242-244`) — correct size, wrong owner.
- **`_reset_idx` responsibility (obstacle resample, grid update, robot pose reset, goal sample, A*, path validation)**: MISSING — `commands.py` has **no `_reset_idx` method at all**. All of that work happens instead in `events.py:reset_map_and_spawn` (`events.py:39-163`), a plain `EventTermCfg` function operating directly on `env`, entirely outside the `CommandTerm`'s lifecycle.
- **`_resample` responsibility (copy path into tracking state, init `path_remaining`)**: INCORRECT — the method exists (`commands.py:50-66`) and attempts the right shape of work, but it reads from the nonexistent `_paths_local` (see above) and calls the stubbed `_compute_path_remaining`.
- **`_update_command` responsibility (waypoint advancement, lookahead recompute, path_remaining update, prev bookkeeping)**: INCORRECT — waypoint-advance logic (`commands.py:71-84`) looks reasonable (distance-to-waypoint threshold, clamp to path length), and `prev_path_remaining` bookkeeping is correctly sequenced *before* recompute (`commands.py:88`), but the two downstream calls (`_compute_lookaheads`, `_compute_path_remaining`, `commands.py:89-90`) are both no-ops.
- **Critical bug check — A* failure zeroes `path_remaining`**: PRESENT, but only at the **env level**, not the `CommandTerm` level. In `events.py:100-105`, `env._path_remaining[env_id]` and `env._prev_path_remaining[env_id]` are explicitly zeroed *before* the `if path is not None:` branch, so a failed A* search correctly leaves them at 0 rather than a stale nonzero value. However, this defensive zeroing is moot for the `CommandTerm`'s own `self.path_remaining`, since that value is never populated at all (stub), regardless of A* success or failure.
- **CommandTerm registered/wired into any env config**: MISSING — grepped the full `source/` tree for `NavigationWaypointCommand`, `CommandsCfg`, `CommandTermCfg`, and `command_manager`; the only hit is the class definition itself. Neither `helixnav_debug_base_env_cfg.py` nor `helixnav_debug_base_rl_env_cfg.py` has a `commands` field, and `manager_configs/__init__.py` never imports anything command-related. **This class is dead code.**

Issues:
- `commands.py:58` — broken reference to `self._env._paths_local`, which is never defined (env only ever has `_paths_world`).
- `commands.py:92-104` — `_compute_lookaheads` and `_compute_path_remaining` are unimplemented stubs.
- No `_reset_idx` exists; the responsibility split mandated by spec is not respected — reset orchestration lives entirely in `events.py`, outside the `CommandTerm`.
- The class is never instantiated by any config in the audited tree.

Required work:
1. Add a `CommandsCfg` (or equivalent) to `helixnav_debug_base_rl_env_cfg.py` and wire `NavigationWaypointCommand` in via a `CommandTermCfg`.
2. Implement `_reset_idx` on the `CommandTerm` to own obstacle resampling, occupancy grid update, robot pose reset, goal sampling, A* execution, and path validation — currently done ad hoc in `events.py:reset_map_and_spawn`. Either migrate that logic into `_reset_idx`, or explicitly document why it stays as an `EventTerm` (deviates from spec either way as currently split).
3. Implement `_compute_lookaheads` and `_compute_path_remaining` for real (currently `pass`).
4. Fix `commands.py:58` to reference the correct path-storage attribute (whatever the migrated reset logic ends up calling it — currently the env-side name is `_paths_world`, not `_paths_local`).
5. Add `goal_positions` and `occupancy_grids` as tensors owned by the `CommandTerm` itself, per spec, instead of leaving them as ad hoc `env._goal_positions` / `env._occupancy_grids` attributes.

---

## Section 2: Observations

**Status: PARTIAL** — depth image and relative-goal terms exist; lookahead vectors and goal-snap flags are entirely absent; base velocity is wired incorrectly; and the one term that does read nav state uses exactly the lazy-init anti-pattern the spec forbids.

Findings:
- **Depth image 96×54×1, D435i-matching 87°×58° FOV**: PRESENT — `manager_configs/scene.py:99-129`, `pattern_cfg=patterns.PinholeCameraPatternCfg(... width=96, height=54 ...)`, `horizontal_aperture=45.55` / `vertical_aperture=26.6068` explicitly commented as 87°/58° FOV (`scene.py:72-74, 104-105`); obs function `mdp.get_depth_images` (`observations.py:21-46`) reads `sensor.data.output["distance_to_camera"]` and normalizes by `max_distance`; wired in `manager_configs/observations.py:51-57` with `sensor_cfg=SceneEntityCfg("depth_camera")`.
- **8 lookahead vectors (unit direction + normalized distance ×8 = 24 dims)**: MISSING — no observation function exists anywhere in `mdp/observations.py`. `manager_configs/observations.py:24-26` has the term **commented out** (`# lookahead_waypoints = ObsTerm(func=mdp....)`, incomplete). Consistent with Section 1: the underlying `CommandTerm.lookaheads` tensor is never computed.
- **8 goal snap flags (8 dims)**: MISSING — no obs function, no `ObsTerm`. The `CommandTerm.snap_flags` tensor exists (`commands.py:39`) but is never written to or read from anywhere.
- **Base velocity, 3 dims (v_x, v_y, yaw_rate)**: INCORRECT — implemented as two *separate, full 3D* terms instead of one filtered 3-dim term: `current_lin_vel = ObsTerm(func=mdp.base_lin_vel)` and `current_ang_vel = ObsTerm(func=mdp.base_ang_vel)` (`manager_configs/observations.py:29-35`). These are isaaclab's stock functions returning full `root_lin_vel_b`/`root_ang_vel_b` (3 dims each = 6 dims total), not a single 3-dim (vx, vy, yaw_rate) vector.
- **Last action, 3 dims**: PRESENT — `prev_actions = ObsTerm(func=mdp.last_action)` (`manager_configs/observations.py:20-22`), matches the 3-dim action space. Note: the attribute `prev_actions` is **defined twice** in the same class body (`observations.py:20-22` and again `38-40`, second passing `mdp.last_action` positionally) — the second silently shadows the first at class-body evaluation time. Functionally harmless (both resolve to the same function) but is dead/duplicate code.
- **Relative goal, 3 dims**: PRESENT (dims correct) but INCORRECT (pattern) — `get_relative_goal_vector` (`observations.py:51-81`) returns `unit_dir (2) + norm_dist (1) = 3`, wired as `rel_goal_w` (`manager_configs/observations.py:43-48`, `arena_size=12.0`). However it reads `env._goal_positions` directly (`observations.py:64`) rather than `env.command_manager.get_term(...)`, and opens with a lazy-init check: `if not hasattr(env, "_nav_state_initialized"): init_nav_state(env)` (`observations.py:59-60`) — this is exactly the anti-pattern the spec prohibits. The file's own header even flags it: `# TODO(cp8): migrate nav state to CommandTerm` (`observations.py:5`).
- **Total scalar MLP input dimensionality = 41**: INCORRECT — cannot be reconciled. Wired scalar terms sum to `3 (prev_actions) + 3 (lin_vel) + 3 (ang_vel) + 3 (rel_goal) = 12` dims, missing the 24 (lookahead) + 8 (snap flags) = 32 dims the spec requires, and even the present terms are 3 dims over target (6 vs 3 for velocity). Additionally, `ObservationsCfg.PolicyCfg.__post_init__` sets `self.concatenate_terms = False` (`manager_configs/observations.py:59-61`), so observations are returned as a **dict**, not a flat 41-dim vector at all — a design choice that may be intentional for a CNN+MLP split architecture, but is worth flagging since the spec speaks of a single flattened dimensionality.
- **All obs functions read from `env.command_manager.get_term(...)`, no lazy-init / module caches**: INCORRECT — violated by `get_relative_goal_vector` (see above). `get_depth_images` doesn't touch nav state at all, so it's clean by default.

Issues:
- No lookahead or goal-snap-flag observation functions exist.
- `get_relative_goal_vector` uses a lazy-init `hasattr` pattern against raw env attributes instead of `command_manager.get_term`.
- Duplicate `prev_actions` term definition in `ObservationsCfg.PolicyCfg`.
- Base velocity observation is 6 dims across two terms, not the 3-dim single term the spec calls for.

Required work:
1. Once `NavigationWaypointCommand` is wired and its compute methods implemented (Section 1), add two new observation functions reading `env.command_manager.get_term("<name>").lookaheads` and `.snap_flags`, and wire them as `ObsTerm`s (24 + 8 dims).
2. Rewrite `get_relative_goal_vector` to source the goal from the command term rather than `env._goal_positions` / lazy `init_nav_state`.
3. Replace the two full base-velocity terms with a single 3-dim (v_x, v_y, yaw_rate) term, or explicitly slice `base_lin_vel[:, :2]` + `base_ang_vel[:, 2:3]` into one function.
4. Remove the duplicate `prev_actions` definition in `manager_configs/observations.py`.
5. Decide and document whether the 41-dim scalar MLP input is meant to be a flat concatenated vector (would require `concatenate_terms=True` or restructuring `PolicyCfg`) or a dict consumed by a multi-branch network.

---

## Section 3: Rewards

**Status: PARTIAL** — only 1 of the 4 required reward terms is implemented; the one that is present is correct.

Findings:
- **Progress (PBRS)**: MISSING — `mdp/rewards.py` (26 lines total) contains only `goal_reached_reward`; no PBRS/progress function exists anywhere in the audited files, and `manager_configs/rewards.py` (15 lines) has only a single `RewTerm`.
- **Goal reached, sparse +50**: PRESENT and correct — `mdp.goal_reached_reward` (`rewards.py:11-26`) returns `(dist <= min_goal_threshold).float()`; wired with `weight=50` and `min_goal_threshold=0.3` (`manager_configs/rewards.py:10-15`), matching the 0.3m termination threshold (see Section 4).
- **Collision: no negative reward**: PRESENT (correctly absent) — no collision-related reward term exists in `mdp/rewards.py` or `manager_configs/rewards.py`. Collision handling is confined to terminations, per spec.
- **Smoothness `-0.01 * ||a_t - a_{t-1}||^2`**: MISSING — no such function exists anywhere in `mdp/rewards.py`.
- **No velocity-scaled reward terms anywhere**: PRESENT (trivially, since almost no reward terms exist yet to violate this).

Issues:
- 2 of 4 required reward terms (progress/PBRS, smoothness) do not exist at all.

Required work:
1. Implement a PBRS progress reward function: `r = gamma * (-path_remaining_new) - (-path_remaining_old)`, reading `path_remaining`/`prev_path_remaining` from the (not-yet-wired) `CommandTerm` — currently impossible to implement correctly since those values are never computed (Section 1).
2. Implement the smoothness penalty `-0.01 * ||a_t - a_{t-1}||^2`, likely against `env.action_manager.action` and the previous step's stored action.
3. Add both to `manager_configs/rewards.py`.

Unexpected additions: none beyond spec in this section.

---

## Section 4: Terminations

**Status: PRESENT** — all three required terms exist and are wired correctly; no critical bug found here (the reward/termination threshold mismatch the spec warns about does **not** occur in this config, values match).

Findings:
- **Goal reached, dist < 0.3m**: PRESENT and correct — `mdp.goal_reached` (`mdp/terminations.py:32-49`) computes `dist <= min_distance_threshold`; wired with `min_distance_threshold=0.3` (`manager_configs/terminations.py:23-29`).
- **Collision, contact sensor on base/legs against obstacles**: PRESENT — `mdp.terminations_by_collisions` (`mdp/terminations.py:13-28`) checks `norm(net_forces_w) > force_threshold` per body, `any` across bodies; wired with `sensor_cfg=SceneEntityCfg("collision_sensor")` (`manager_configs/terminations.py:14-20`). (Underlying sensor regex has a separate bug — see Section 6.)
- **Timeout**: PRESENT — `time_out = DoneTerm(func=mdp.time_out, time_out=True)` (`manager_configs/terminations.py:32-35`). `mdp.time_out` isn't defined locally but resolves via `mdp/__init__.py:1` (`from isaaclab.envs.mdp import *`), isaaclab's stock timeout termination — not a broken reference.
- **Critical bug check — reward-side goal threshold not wider than termination threshold**: PRESENT/correct in this wired config — both `manager_configs/rewards.py:13` and `manager_configs/terminations.py:26` use `0.3`. **However**, the function *defaults* differ: `mdp/rewards.py:14` defaults `min_goal_threshold=0.5`, `mdp/terminations.py:35` defaults `min_distance_threshold=0.5`. Any future env config that omits these `params` overrides would silently regress to a wider reward-payout radius than a *0.3m-hardcoded* expectation elsewhere, or — more subtly — if only one of the two configs is left at its default while the other is overridden, the two would diverge and reintroduce exactly the reward-farming bug the spec warns about. Flagging as a latent/dormant risk, not an active bug in the audited debug config.

Issues:
- None active in the wired debug config. Latent risk: default threshold values in `mdp/rewards.py` and `mdp/terminations.py` (0.5m) are inconsistent with the spec's 0.3m and with each other's config-level overrides, relying on both config sites remembering to override consistently.

Required work:
1. Consider changing the function defaults in `mdp/rewards.py:14` and `mdp/terminations.py:35` to `0.3` to match spec and remove the reliance on both call sites getting the override right.

Unexpected additions: none.

---

## Section 5: Events / Reset ordering

**Status: PARTIAL** — all the individual pieces of work happen, but the actual sequence deviates from the specified order, occupancy-grid population is analytic rather than sensor-driven, and path length is never validated (only existence).

Findings:
- **1. Sample obstacles**: PRESENT — `events.py:60-93`, obstacle positions from `map_spec.obstacles` written into `env._obstacles_pos`.
- **2. Update occupancy grid**: INCORRECT — `env._occupancy_grids[env_id]` (`events.py:91-93`) is populated directly from `map_spec.occupancy_grid`, an analytically-generated grid from `RandomMapGenerator`, computed *before* obstacles are ever written to the sim (`_spawn_obstacles` happens later, `events.py:142`). The actual physical `occupancy_scanner` `MultiMeshRayCaster` sensor (`scene.py:582-607`) is read only for optional debug visualization (`events.py:148-160`), never to populate the grid used for A*/policy. This isn't necessarily wrong (an analytic grid can be a valid design), but it means "update occupancy grid" as a distinct sim-observing step doesn't happen the way the spec's ordering implies — it's precomputed math, not a scan.
- **3. Reset robot pose (start position + yaw)**: PRESENT, but out of order — `_spawn_robot` (`events.py:183-220`) is called at `events.py:145`, **after** obstacle spawning and, critically, after goal sampling and A* have already run inside the per-env loop at `events.py:60-139`. Spec requires robot pose reset (step 3) to precede goal sampling (step 4) and A* (step 5); actual order computes goal+A* first, then resets pose.
- **4. Sample goal position (validate reachability)**: PRESENT for sampling (`events.py:86-88`), but INCORRECT for the "validate reachability" clause — there is no distinct reachability check; reachability is only implicitly established by whether A* subsequently succeeds (`events.py:97`), which conflates steps 4 and 5-6 rather than validating goal reachability as its own gate before running A*.
- **5. Run A***: PRESENT — `events.py:96-97`, `astar(inflated, map_spec.start_grid, map_spec.goal_grid)` on a 1-cell-inflated grid.
- **6. Validate path exists and is above minimum length**: INCORRECT — only existence is checked (`events.py:107`, `if path is not None:`). No minimum-length validation exists anywhere in `events.py`.
- **7. Compute lookaheads**: MISSING — never called from the event function at all; the only place lookaheads would be computed is `CommandTerm._compute_lookaheads`, which is both a stub (Section 1) and never invoked (the `CommandTerm` isn't wired into any config).
- **8. First observation generated only after all of the above**: Cannot be fully verified as designed, since step 7 never happens. For the steps that do run, they're synchronous within the single `reset_map_and_spawn` event function body, so isaaclab's manager-based reset flow (compute obs after all `mode="reset"` events complete) should hold for what's actually implemented — there's no async/deferred computation observed.

Issues:
- Actual order is: (obstacle+grid+goal+A* all computed analytically in Python, per-env, `events.py:60-139`) → obstacles written to sim (`events.py:142`) → robot pose reset (`events.py:145`) → zero prev actions (`events.py:163`). This does not match the spec's literal 1-2-3-4-5-6-7 sequence (robot pose reset is supposed to precede goal sampling and A*).
- No minimum path-length validation.
- Lookahead computation never happens as part of the reset flow.

Required work:
1. Either re-order the reset event to match the spec's sequence (robot pose before goal/A*) or explicitly document that the analytic map-generation approach makes that ordering constraint moot (goal/A* don't depend on physical robot placement) — as currently written this is undocumented and looks accidental given the file's own step-numbered docstring (`events.py:48-53`) doesn't match its own code order.
2. Add a minimum path-length check after A* succeeds; treat too-short paths the same as A* failure (zero `path_remaining`, do not spawn a valid episode).
3. Call lookahead computation as part of this reset flow once implemented (Section 1).

Unexpected additions: `map_spec.occupancy_grid` (analytic, from `RandomMapGenerator`) as the source of truth for the grid used in A*/policy, rather than the physical `occupancy_scanner` sensor — not in spec's literal wording, noting without judgment.

---

## Section 6: Collision sensor

**Status: PARTIAL** — spawner-level flag is correctly placed; the regex has the exact operator-precedence bug the spec calls out as a known audit item.

Findings:
- **`activate_contact_sensors=True` on the spawner, not the sensor config**: PRESENT and correct — `isaaclab_assets/robots/unitree.py:143`, set directly on `UsdFileCfg` (the robot's spawner), as a sibling field to `rigid_props=sim_utils.RigidBodyPropertiesCfg(...)` (`unitree.py:144-152`) — not nested inside `RigidBodyPropertiesCfg` itself (isaaclab's actual API places this field on the spawner cfg, not on `RigidBodyPropertiesCfg`), and not set anywhere on `ContactSensorCfg` (`scene.py:132-135`), which has no such field configured. Matches the spirit of the check.
- **Contact sensor regex operator precedence**: INCORRECT — **critical bug, confirmed**. `manager_configs/scene.py:133`: `prim_path="{ENV_REGEX_NS}/robot/.*_(hip|thigh)|Head_(upper|lower)"`. Regex alternation (`|`) has the lowest precedence, so this parses as `({ENV_REGEX_NS}/robot/.*_(hip|thigh))` **OR** `(Head_(upper|lower))` — the second alternative has **no `{ENV_REGEX_NS}/robot/` prefix at all**. Since every real prim path is absolute (e.g. `/World/envs/env_0/robot/...`), the bare `Head_upper`/`Head_lower` alternative can never match an actual prim path under the robot. This is exactly the known bug class the spec warns about — the sensor is very likely silently blind to head-body contacts (whatever `Head_upper`/`Head_lower` were meant to catch), while `hip`/`thigh` contacts under the robot prefix do work.
- **No many-to-many filtering attempted**: PRESENT and correct — `ContactSensorCfg` at `scene.py:132-135` sets no `filter_prim_paths_expr` or similar; only `prim_path` and `update_period` are configured.

Issues:
- `manager_configs/scene.py:133` — regex precedence bug, second alternative unreachable.

Required work:
1. Wrap the alternation: `prim_path="{ENV_REGEX_NS}/robot/(.*_(hip|thigh)|Head_(upper|lower))"`, or fully qualify each branch: `{ENV_REGEX_NS}/robot/.*_(hip|thigh|Head_upper|Head_lower)` (adjust to match actual prim naming).

---

## Section 7: Scene / Assets

**Status: PARTIAL** — obstacle placement architecture is correct; shape-type choice is wrong across the board and would silently break raycasting against every obstacle and wall.

Findings:
- **Obstacles as `RigidObjectCfg`, defined in the env config, not baked into the USD scene**: PRESENT and correct — `s_obs_0` through `s_obs_12` (`manager_configs/scene.py:286-550`) are all `RigidObjectCfg` instances defined directly in `HelixNavDebugBaseScene`, under `{ENV_REGEX_NS}/static_obstacles/s_obs_N` (a dummy parent prim `static_obstacles` created at `scene.py:229-232`). Not baked into a static USD stage.
- **Obstacles use `MeshCuboidCfg`, not `CuboidCfg`**: INCORRECT — **critical bug, confirmed**. Every single obstacle (`s_obs_0`...`s_obs_12`) and every wall (`wall_north/south/east/west`) uses `sim_utils.CuboidCfg` (e.g. `scene.py:288, 143`), never `MeshCuboidCfg`. The only prim in the whole scene config using `MeshCuboidCfg` is the invisible `occupancy_scanner_mount` (`scene.py:568`), which isn't itself an obstacle. Per spec, `CuboidCfg` is invalid for raycasting — and both raycast-dependent sensors target exactly these prims: `depth_camera`'s `mesh_prim_paths` includes `{ENV_REGEX_NS}/static_obstacles/s_obs.*` and the four walls (`scene.py:114-124`), and `occupancy_scanner`'s `mesh_prim_paths` targets the same obstacle/wall prims (`scene.py:584-594`). If `CuboidCfg` truly can't be hit by these raycasters, both the policy's depth image and the occupancy grid used for A* could be systematically blind to every obstacle and wall in the scene — a severe, silent failure mode. (Note: cylinders/cones also use `sim_utils.CylinderCfg`/`ConeCfg`, `scene.py:408-550` — spec only calls out cuboid-vs-mesh-cuboid, but the same raycasting concern would apply to whatever the equivalent primitive-vs-mesh split is for those shapes; flagging cuboids per the literal spec item.)
- **Depth sensor `update_period=1e6` if lazy-eval occupancy scanner variant, else standard per-step**: PRESENT with a minor value mismatch — `depth_camera` (per-step CNN-branch sensor) correctly uses `update_period=0.05` (20Hz, standard per-step, `scene.py:101`). `occupancy_scanner` (the lazy-eval variant used to build the grid) uses `update_period=999.0` (`scene.py:600`, commented "effectively never auto-updates") rather than the spec's literal `1e6`. Functionally equivalent intent (large value to suppress auto-updates within any realistic episode), but the exact constant differs from spec.

Issues:
- All obstacle and wall spawn configs use `CuboidCfg`; per spec this is invalid for raycasting, and both raycast sensors in the scene target these exact prims.

Required work:
1. Change `s_obs_0`...`s_obs_12` and `wall_north/south/east/west` from `sim_utils.CuboidCfg` to `sim_utils.MeshCuboidCfg` (verify the isaaclab raycaster documentation/behavior confirms primitive `CuboidCfg` prims are in fact unhittable before changing — this audit reports the spec's stated requirement, not independently-verified raycaster internals).
2. Confirm whether cylinder/cone obstacles need an equivalent mesh-backed variant for raycasting to hit them.
3. Optionally align `occupancy_scanner.update_period` to the spec's literal `1e6` for consistency, though `999.0` is very likely functionally fine.

Unexpected additions: RGB camera (`rgb_camera`, `scene.py:78-96`) and `height_scanner` (`scene.py:59-66`, inherited pattern for the locomotion policy) exist in the scene but aren't part of the CP7 spec items — noting without judgment, height_scanner is presumably required by the frozen low-level locomotion policy's own observation space (see Section 8), not part of the nav-policy observation set.

---

## Section 8: Action space wiring

**Status: PARTIAL** — action dimensionality and semantics (dim 2 = yaw_rate) are correct; the checkpoint path doesn't match what the spec names; and output clipping before the low-level policy is entirely absent.

Findings:
- **Frozen locomotion policy consumes 3-dim (v_x, v_y, yaw_rate), dim 2 = yaw_rate**: PRESENT and correct — `PreTrainedPolicyAction.action_dim` is hardcoded to `3` (`mdp/pre_trained_policy_action.py:71-73`). Corroborated by both debug scripts, which build `nav_command = torch.zeros(env.num_envs, 3, ...)` and explicitly assign `nav_command[:, 2] = yaw_rate` from a `Se2Keyboard` controller (`config/go2/env_configs/debug/keyboard_controller_debugger.py:78-80`, `rl_env_keyboard_controller.py:79-82`), and `manager_env.py:55-57` similarly comments `# [vx, vy, yaw_rate]`.
- **`PreTrainedPolicyActionCfg` wraps the checkpoint at `logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt`**: INCORRECT / unverifiable as specified — `manager_configs/actions.py:16-19` resolves `policy_path` to a locally packaged file: `_LOCO_PT = _GO2_CONFIG_DIR / "policies" / "locomotion" / "policy.pt"` (confirmed to exist on disk at `config/go2/policies/locomotion/policy.pt`). Grepped the entire repository for the string `2026-06-13_19-33-23` and `model_1499` — **zero matches anywhere**. There is no evidence in the audited files that `policies/locomotion/policy.pt` is derived from, or traceable to, the specified checkpoint. It may well be a manually-copied renamed file, but that provenance isn't recorded anywhere in-repo.
- **Nav policy output clipped/squashed to roughly ±1.0 before feeding the locomotion layer**: MISSING — `process_actions` (`mdp/pre_trained_policy_action.py:87-88`) is a direct passthrough: `self._raw_actions[:] = actions`. No `torch.clamp`, `tanh`, or any bounding operation appears anywhere in `PreTrainedPolicyAction`. `raw_actions` is fed straight through as `processed_actions` (`pre_trained_policy_action.py:79-81`) and ultimately becomes `self._raw_actions` — the same tensor passed as `velocity_commands` to the low-level policy's observation (`actions.py`... via `cfg.low_level_observations.velocity_commands.func`, `pre_trained_policy_action.py:59`) with no bound.

Issues:
- No clipping/squashing anywhere in `PreTrainedPolicyAction.process_actions`.
- Checkpoint provenance for `policies/locomotion/policy.pt` is untraceable to the path named in spec.

Required work:
1. Add clamping (or `tanh`) to `process_actions` in `mdp/pre_trained_policy_action.py`, bounding `self._raw_actions` to roughly `[-1, 1]` before it's consumed by the low-level observation manager.
2. Confirm out-of-band whether `config/go2/policies/locomotion/policy.pt` is in fact a copy of `logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt`; if so, consider recording that provenance (e.g. a comment or a checked-in manifest) since nothing in the repo currently documents it.

Unexpected additions: none.

---

## Critical bugs (prioritized)

1. **`NavigationWaypointCommand` is never wired into any env config** (`commands.py`, no `CommandsCfg` anywhere) — the entire CommandTerm architecture the spec mandates is dead code; all live nav state instead lives in ad hoc `env.*` attributes set by `events.py`. This is the root cause behind most of the Section 1/2/5 gaps.
2. **`_compute_lookaheads` and `_compute_path_remaining` are unimplemented stubs** (`commands.py:92-104`, literally `pass`) — even if the CommandTerm were wired in, lookaheads and path_remaining would never be computed, so PBRS reward and lookahead/snap-flag observations would be permanently zero.
3. **`commands.py:58` references `self._env._paths_local`, which does not exist anywhere in the codebase** — confirmed via full-repo grep (only hit is this line; the actual env attribute is `_paths_world`). Would raise `AttributeError` the instant `_resample` runs.
4. **All obstacle and wall prims use `sim_utils.CuboidCfg` instead of `MeshCuboidCfg`** (`manager_configs/scene.py`, ~17 spawn configs) — per spec this is invalid for raycasting, and both `depth_camera` and `occupancy_scanner` raycast against exactly these prims. Risk of depth image and A*/occupancy grid being silently blind to every obstacle and wall.
5. **Contact sensor regex operator-precedence bug** (`manager_configs/scene.py:133`: `.*_(hip|thigh)|Head_(upper|lower)`) — the exact known audit item from the spec; the second alternative loses its path prefix and can never match a real prim.
6. **Nav policy output is never clipped/squashed** before being fed to the frozen locomotion policy (`mdp/pre_trained_policy_action.py:87-88`) — spec requires ~±1.0 bounding; none exists.
7. **Reward/termination goal-threshold function defaults diverge from spec and from each other** (`mdp/rewards.py:14` defaults `0.5`, `mdp/terminations.py:35` defaults `0.5`, spec wants `0.3`) — dormant in the currently-wired debug config (both explicitly overridden to `0.3`), but a latent trap for any future config that forgets to override one side.
8. **No minimum path-length validation after A***, only existence is checked (`events.py:107`) — a degenerate 1-2 point "path" could pass through as valid.
9. **Duplicate `prev_actions` `ObsTerm` definition** in `manager_configs/observations.py:20-22` / `38-40` — silently shadowed, harmless today but a sign the file wasn't fully cleaned up when other terms were stubbed out; also, `depth_images` is referenced as an undefined local variable in both `keyboard_controller_debugger.py:110` and `rl_env_keyboard_controller.py:118` inside the (default-off) `--save-camera-images` branch — would raise `NameError` if ever exercised.

---

## Recommended implementation order

Given that the `CommandTerm` layer is unwired and partially stubbed, and several other pieces (rewards, obstacle raycasting) are foundational to everything built on top, the safest build order is:

1. **Fix the raycasting foundation first** (Section 7, critical bug 4): switch obstacle/wall spawns to `MeshCuboidCfg`. Everything downstream (occupancy grid, depth image, A*) depends on this being correct, and it's cheap to verify in isolation (spawn a scene, check raycaster hits).
2. **Fix the contact sensor regex** (Section 6, critical bug 5): small, isolated, and terminations/rewards correctness depends on it not silently missing collisions.
3. **Build out `NavigationWaypointCommand` properly** (Section 1): implement `_reset_idx` (migrating/absorbing the logic currently in `events.py:reset_map_and_spawn`), implement `_compute_lookaheads` and `_compute_path_remaining` for real, fix the `_paths_local` reference, add `goal_positions`/`occupancy_grids` as CommandTerm-owned tensors, add minimum path-length validation. This is the largest single chunk of missing work and blocks Sections 2, 3, and 5.
4. **Wire the CommandTerm into the env config** (`CommandsCfg` in `helixnav_debug_base_rl_env_cfg.py`) once step 3 is functionally complete.
5. **Rewrite observations** (Section 2) to read from `command_manager.get_term(...)`: fix `get_relative_goal_vector`, add lookahead-vector and goal-snap-flag observation functions, collapse base velocity into a single 3-dim term, remove the duplicate `prev_actions` term, and decide on flat-vs-dict obs shape.
6. **Implement the missing reward terms** (Section 3): PBRS progress (now possible once `path_remaining`/`prev_path_remaining` are real) and smoothness. Goal-reached is already correct and can stay as-is.
7. **Add action clipping** (Section 8, critical bug 6) to `PreTrainedPolicyAction.process_actions` — small, isolated, no dependency on the above.
8. **Clean up dormant/latent issues**: align reward/termination threshold defaults (Section 4/critical bug 7), remove the duplicate obs term and fix the debug-script `NameError` (critical bug 9), confirm/document the locomotion checkpoint provenance (Section 8).
