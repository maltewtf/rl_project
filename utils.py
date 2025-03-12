import numpy as np
import random 

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

def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(-1, 1) # choose random action
    else:
        # returns the key of the maximum value in the dictionary 
        return max(Q[state], key=Q[state].get) if state in Q else 0

def reduce_Q_to_V(Q, game):
    """reduces Q to V by choosing the action with the highes Q value"""
    V = {}
    for state in [(y, x) for y in range(game.height) for x in range(game.width)]:
        V[state] = max(Q[state], key=Q[state].get) if state in Q else 0

    return V
        