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

            state, reward, done = env.step(state, action)

        if reward > 0:
            success += 1
        elif count_partial_success:
            success += state[0] / env.height

    if not silent:
        # print(f"level completion: {success:.2f}/{len(starts)}")
        print(f"completion rate: {success/len(starts):.2f}")

    return success / len(starts)

def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(-1, 1) # choose random action
    else:
        # returns the key of the maximum value in the dictionary 
        # return max(Q[state], key=Q[state].get) if state in Q else 0
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
    """returns the key with the highest associated value (sadly the np.argmax function does not work on dictionaries)"""
    return max(d, key=d.get) if len(d) > 0 else default

# Moved from Dymanic_Programmming.py, dunno if we will use in the end.
def simulate_agent(env, policy):
    """Simulates an agent playing the game using the learned policy."""
    state = env.reset()  # Start at the initial position
    total_reward = 0
    steps = 0

    print("\nAgent Simulation:\n")

    while True:
        env.print_state(state)

        if state in policy:
            best_action = policy[state]
        else:
            print(f"State {state} not in learned policy; defaulting to STAY.")
            best_action = env.actions.index(0)

        action = best_action
        print(f"Step {steps}: Agent at {state}, taking action {action}")

        next_state, reward, done = env.step(state, action)
        total_reward += reward
        steps += 1

        if done:
            print(f"\nGame Over! Final State: {next_state}, Total Reward: {total_reward}, Steps Taken: {steps}\n")
            env.print_state(next_state)
            break

        state = next_state