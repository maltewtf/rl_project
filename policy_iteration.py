from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = defaultdict(float)

    while True:
        delta = 0
        for state in policy:

            v = 0  # Expected return for this state

            for action_idx, action_prob in enumerate(policy[state]):
                action = env.actions[action_idx]
                next_state, reward, done = env.get_next_state(state, action)

                if done:
                    v += action_prob * reward  # If terminal, use reward directly
                else:
                    v += action_prob * (reward + gamma * V[next_state])

            delta = max(delta, abs(V[state] - v))
            V[state] = v

        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    """Improves policy by choosing the greedy action based on V(s)."""
    policy = {}
    states, actions = env.get_states_actions()

    for state in states:
        Q_values = np.zeros(len(actions))  # Store Q-values for all actions

        for action_idx, action in enumerate(actions):
            next_state, reward, done = env.get_next_state(state, action)

            # Compute Q-values correctly
            Q_values[action_idx] = reward + gamma * V[next_state] * (not done)

        # Select the best action greedily
        best_action = np.argmax(Q_values)
        policy[state] = np.eye(len(actions))[best_action]  # One-hot encoding

    return policy


def policy_iteration(env, gamma=0.99, theta=1e-6):
    """Finds the optimal policy using Policy Iteration."""
    states, actions = env.get_states_actions()

    # Initialize a random policy (equal probability for all actions)
    policy = {state: np.ones(len(actions)) / len(actions) for state in states}

    while True:
        V = policy_evaluation(env, policy, gamma, theta)  # Evaluate policy
        new_policy = policy_improvement(env, V, gamma)  # Improve policy

        # Check convergence
        if all(np.array_equal(policy[s], new_policy[s]) for s in policy):
            break

        policy = new_policy  # Update policy

    return policy, V

def simulate_agent(env, policy):
    """Simulates an agent playing the game using the learned policy."""
    state = env.reset()  # Start at initial position
    total_reward = 0
    steps = 0

    print("\nAgent Simulation:\n")

    while True:
        env.print_state(state)

        if state in policy:
            best_action = np.argmax(policy[state])
        else:
            print(f"State {state} not in learned policy; defaulting to STAY.")
            best_action = env.actions.index(0)

        action = env.actions[best_action]
        print(f"Step {steps}: Agent at {state}, taking action {action}")

        next_state, reward, done = env.get_next_state(state, action)
        total_reward += reward
        steps += 1

        if done:
            print(f"\nGame Over! Final State: {next_state}, Total Reward: {total_reward}, Steps Taken: {steps}\n")
            env.print_state(next_state)
            break

        state = next_state

def inspect_policy(policy, env):
    """Prints the learned policy's action probabilities for each state."""
    print("\n--- Policy Inspection ---\n")
    action_names = {env.LEFT: "←", env.STAY: "•", env.RIGHT: "→"}

    for state in sorted(policy.keys()):
        action_probs = policy[state]  # Action probabilities
        action_str = ", ".join(
            [f"{action_names[env.actions[i]]}: {p:.2f}" for i, p in enumerate(action_probs)]
        )
        print(f"State {state}: {action_str}")

    print("\n")

def inspect_value_function(V):
    """Prints the state value function to understand learned state values."""
    print("\n--- Value Function Inspection ---\n")

    for state in sorted(V.keys()):
        print(f"State {state}: V(s) = {V[state]:.2f}")

    print("\n")

def print_policy_grid(policy, env):
    """Prints the policy as a visual grid."""
    grid = np.full((env.height, env.width), " ", dtype=str)
    action_symbols = {env.LEFT: "←", env.STAY: "•", env.RIGHT: "→"}

    for (y, x) in policy:
        if env.level[y][x] == 1:
            grid[y, x] = "X"  # Obstacle
        else:
            best_action = np.argmax(policy[(y, x)])  # Get the best action
            grid[y, x] = action_symbols[env.actions[best_action]]

    print("\n--- Policy Grid ---\n")
    for y in range(env.height):
        print(" ".join(grid[y]))
    print("\n")

def value_iteration(env, gamma=0.99, theta=1e-6):
    """Finds the optimal policy using Value Iteration."""
    V = defaultdict(float)
    states, actions = env.get_states_actions()

    while True:
        delta = 0

        for state in states:
            v = V[state]  # Expected return for this state

            q_values = np.zeros(len(actions))  # Store Q-values for all actions

            for action_idx, action in enumerate(actions):
                next_state, reward, done = env.get_next_state(state, action)

                # Compute Q-values correctly
                q_values[action_idx] = reward + gamma * V[next_state] * (not done)

            V[state] = max(q_values)  # Update state value

            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    # Extract the optimal policy
    policy = {}
    for state in states:
        q_values = np.zeros(len(actions))
        for action_idx, action in enumerate(actions):
            next_state, reward, done = env.get_next_state(state, action)
            q_values[action_idx] = reward + gamma * V[next_state] * (not done)
        best_action = np.argmax(q_values)
        policy[state] = np.eye(len(actions))[best_action]  # One-hot encoding

    return policy, V