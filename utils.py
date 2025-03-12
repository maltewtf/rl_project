import numpy as np

def print_V(policy, env):
    """Prints the policy as a visual grid."""
    grid = np.full((env.height, env.width), " ", dtype=str)
    action_symbols = {env.LEFT: "←", env.STAY: "•", env.RIGHT: "→"}

    for (y, x) in policy:
        if env.level[y][x] == 1:
            # grid[y, x] = "_"  # Obstacle
            grid[y, x] = "\u2586"  # Obstacle
        else:
            grid[y, x] = action_symbols[policy[(y, x)]]

    print("\n--- Policy Grid ---\n")
    for y in range(env.height):
        print(" ".join(grid[y]))
    print("\n")

def test_policy(policy, env, test_all_starts=True):
    """runs the game with a given policy and then returns a success ratio"""
    
    if test_all_starts:
        starts = range(env.width)
    else:
        starts = [env.width // 2]

    success = 0
    reward = 0

    for start_x in starts:
        done = False
        state = env.reset(start_x)

        while not done:        
            state, reward, done = env.get_next_state(state, policy[state])

        if reward > 0:
            success += 1

    print(f"level completion: {success}/{len(starts)}")
    return success/len(starts)

        