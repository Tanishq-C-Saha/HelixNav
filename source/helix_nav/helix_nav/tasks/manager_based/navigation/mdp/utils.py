from isaaclab.envs import ManagerBasedEnv
from isaaclab.sim import get_current_stage


# helper function to get child prim names 
def get_static_obstacles(env: ManagerBasedEnv):
    """
    Return all obstacle child prims under the static_obstacles parent.
    """

    stage = get_current_stage()

    parent_path = "/World/envs/env_0/static_obstacles"
    parent_prim = stage.GetPrimAtPath(parent_path)

    if not parent_prim.IsValid():
        raise RuntimeError(
            f"Could not find prim at {parent_path}"
        )

    obstacles = []

    for child in parent_prim.GetChildren():
        print(f"[DEBUG]: child: {child}")
        print(f"[DEBUG]: child_name : {child.GetName()}")
        if child.GetName().startswith("s_obs_"):
            obstacles.append(child)

    return obstacles


# returns list of tuples of obstacle sizes 
def get_obstacle_pool(env: ManagerBasedEnv) -> dict:
    """
    Uses scene to get all the available static obstacles for now,
    Return the sizes of all available static obstacles.
    #! for dynamic obsatcles : make additions to the function
    """
    static_obstacles = []
    static_obstacles_pool = []
    # returns static obstacles from stage
    # NOTE: only obstacles under "{ENV_REGEX_NS}/static_obstacles/" are returned
    static_obstacles_prim = get_static_obstacles(env=env)

    # get prim name -> add in obstacle pool

    for obs_prim in static_obstacles_prim:
        obs = env.scene[obs_prim.GetName()]
        static_obstacles.append(obs)

        # Get size for THIS obstacle only — no inner loop
        if hasattr(obs.cfg.spawn, "size"):
            static_obstacles_pool.append(obs.cfg.spawn.size)
        elif hasattr(obs.cfg.spawn, "radius"):
            size = (
                obs.cfg.spawn.radius,
                obs.cfg.spawn.radius,
                obs.cfg.spawn.height,
            )
            static_obstacles_pool.append(size)

    print(f"[DEBUG]: Obstacle pool = {static_obstacles_pool}\n")

    # NOTE: In future this function can be modified to provide dynamic obstacle pool
    return {
        "static_obstacles_pool": static_obstacles_pool,
        "static_obstacles": static_obstacles   # entity in the scene #! not prim 
    }

