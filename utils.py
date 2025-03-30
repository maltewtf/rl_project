import numpy as np
import random
from collections import defaultdict


def print_policy(policy, env):
    """Prints the policy as a visual grid."""
    grid = np.full((env.height, env.width), " ", dtype=str)
    action_symbols = {env.LEFT: "←", env.STAY: "•", env.RIGHT: "→"}

    for (y, x) in policy:
        if env.level[y][x] == 1:
            grid[y, x] = "\u2586"  # Obstacle
        else:
            action_probabilities = policy[(y, x)]

            # Check if policy is an array and get the best action
            if isinstance(action_probabilities, (np.ndarray, list)):
                best_action_index = np.argmax(action_probabilities)
                action = env.actions[best_action_index]
            else:
                # Fallback if policy directly contains action integers
                action = action_probabilities

            grid[y, x] = action_symbols[action]

    print("\n--- Policy Grid ---\n")
    for y in range(env.height):
        print(" ".join(grid[y]))
    print("\n")

def test_policy(policy, env, test_all_starts=True, silent=False, count_partial_success=False):
    """
    Runs the game with a given policy and returns a success ratio.

    Parameters:
        test_all_starts: When True, tests all possible starting positions and returns the success rate.
        silent: When True, doesn't print anything at the end.
        count_partial_success: When True, counts partial success based on how far the agent progressed.
    """

    if test_all_starts:
        starts = range(env.width)
    else:
        starts = [env.width // 2]

    success = 0

    for start_x in starts:
        done = False
        state = env.reset(start_x)

        while not done:
            if state in policy:
                action_probs = policy[state]
                if isinstance(action_probs, (np.ndarray, list)):
                    action_index = np.argmax(action_probs)
                    action = env.actions[action_index]
                else:
                    action = action_probs
            else:
                action = env.STAY  # default action if state is missing

            state, reward, done = env.get_next_state(state, action)

        if reward > 0:
            success += 1
        elif count_partial_success:
            success += state[0] / env.height

    if not silent:
        print(f"level completion: {success}/{len(starts)}")

    return success / len(starts)

def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(-1, 1) # choose random action
    else:
        # returns the key of the maximum value in the dictionary 
        return max(Q[state], key=Q[state].get) if state in Q else 0
    
def n_epsilon_greedy_policy(Qs, state, epsilon):
    """applies epsilon greedy strategy to the sum of multiple Qs"""
    if random.uniform(0, 1) < epsilon:
        return random.randint(-1, 1) # choose random action
    else:
        summed_Q = defaultdict(lambda: 0)
        for actions in [Q[state] for Q in Qs]:
            for key in actions.keys():
                summed_Q[key] += actions[key]

        return max(summed_Q[state], key=summed_Q[state].get) if state in summed_Q else 0
    
def sum_Q(Qs):
    """returns the sum of Q-values of a list of state action spaces"""
    summed_Q = defaultdict(lambda: defaultdict(lambda: 0))

    for Q in Qs:
        for state in Q.keys():
            for action in Q[state].keys():
                summed_Q[state][action] += Q[state][action]

    return summed_Q


def Q_to_policy(Q, game):
    """Converts Q-values into a greedy one-hot encoded policy compatible with inspect_policy."""
    policy = {}
    num_actions = len(game.actions)

    for state, action_dict in Q.items():
        q_values_array = np.array([action_dict[a] for a in game.actions])
        best_action_index = int(np.argmax(q_values_array))
        one_hot_policy = np.zeros(num_actions)
        one_hot_policy[best_action_index] = 1.0
        policy[state] = one_hot_policy

    return policy

def print_V(V, env):
    """Prints the value function as a visual grid."""
    grid = np.zeros((env.height, env.width), dtype=float)

    for (y, x), value in V.items():
        if env.level[y][x] == 1:
            grid[y, x] = np.nan  # obstacle
        else:
            grid[y, x] = value

    print("\n--- Optimal V (Value Function) Grid ---\n")
    for row in grid:
        print(" ".join(["{:6.2f}".format(v) if not np.isnan(v) else "  X   " for v in row]))
    print("\n")
        