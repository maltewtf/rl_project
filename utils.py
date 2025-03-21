import numpy as np
import random
from collections import defaultdict

def print_policy(policy, env):
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

def test_policy(policy, env, test_all_starts=True, silent=False, count_partial_success=False):
    """
    runs the game with a given policy and then returns a success ratio.

    Parameters:
        test_all_starts: when True, test all possible starting possition and return the passrate of all together.
        silent: when True, dont print anything at the end.
        count_partial_success: when True, in addion to adding 1 for completing the level also add the achieved player hight/level height on death to the pass rate.
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
            state, reward, done = env.get_next_state(state, policy[state])

        if reward > 0:
            success += 1
        elif count_partial_success:
            success += state[0] / env.height

    if not silent:
        print(f"level completion: {success}/{len(starts)}")

    return success/(len(starts))

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
    """reduces Q to V by choosing the action with the highes Q value"""
    V = {}
    for state in [(y, x) for y in range(game.height) for x in range(game.width)]:
        V[state] = max(Q[state], key=Q[state].get) if state in Q and len(Q[state]) > 0 else 0

    return V

        