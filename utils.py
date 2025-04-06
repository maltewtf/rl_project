import numpy as np
import random
from collections import defaultdict
from game import MDPGame

class Types:
    """
    defines data types/structures and constructors for Q, V, and Policy. 
    Makes the functions a little more compact and foolproof. (also avoids namespace problems by having Types in front)

    "__new__(self)" just replaces the constructor
    """
    class Q:
        def __new__(self):
            return defaultdict(lambda : defaultdict(float))

    class V:
        def __new__(self):
            return defaultdict(float)
        
    class Policy:
        def __new__(self):
            return defaultdict(int)
        
    class Returns:
        def __new__(self):
            return defaultdict(list)


def print_policy(policy: Types.Policy, env: MDPGame):
    """Prints the policy as a visual grid."""
    grid = np.full((env.height, env.width), " ", dtype=str)
    action_symbols = {env.LEFT: "←", env.STAY: "•", env.RIGHT: "→"}

    for (y, x) in [(y, x) for x in range(env.width) for y in range(env.height)]:
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
            state, reward, done = env.step(state, policy[state])

        if reward > 0:
            success += 1
        elif count_partial_success:
            success += state[0] / env.height

    if not silent:
        print(f"level completion: {success}/{len(starts)}")

    return success/(len(starts))

def epsilon_greedy_policy(Q, state, epsilon):
    """applies epsilon greedy strategy to a given state of Q using epsilon and returns the chosen action/key of Q"""
    if random.uniform(0, 1) < epsilon:
        return random.randint(-1, 1) # choose random action
    else:
        return argmax(Q[state])
    
def n_epsilon_greedy_policy(Qs, state, epsilon):
    """applies epsilon greedy strategy to the sum of multiple Qs"""
    if random.uniform(0, 1) < epsilon:
        return random.randint(-1, 1) # choose random action
    else:
        summed_action_values = defaultdict(lambda: 0)
        for action_values in [Q[state] for Q in Qs]: 
            for action in action_values.keys():
                summed_action_values[action] += action_values[action]
        return max(summed_action_values, key=summed_action_values.get) if len(action_values) > 0 else 0
    
def sum_Q(Qs):
    """returns the sum of Q-values of a list of state action spaces"""
    summed_Q = defaultdict(lambda: defaultdict(lambda: 0))

    for Q in Qs:
        for state in Q.keys():
            for action in Q[state].keys():
                summed_Q[state][action] += Q[state][action]

    return summed_Q

def Q_to_policy(Q, env):
    """reduces Q to V by choosing the action with the highes Q value"""
    policy = {}
    for state in [(y, x) for y in range(env.height) for x in range(env.width)]:
        policy[state] = max(Q[state], key=Q[state].get) if state in Q and len(Q[state]) > 0 else 0

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
        print(" ".join([f"{v:6.2f}" if not np.isnan(v) else "  X   " for v in row]))
    print("\n")
        
def argmax(d, default=0):
    """
    takes a dictionary whith numerals as values
    returns the key with the highest associated value (sadly the np.argmax function does not work on dictionaries)
    """
    return max(d, key=d.get) if len(d) > 0 else default