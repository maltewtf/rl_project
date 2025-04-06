import numpy as np
from collections import defaultdict
from game import *
from utils import *

from collections import defaultdict
import numpy as np

def mc_prediction(env, policy, gamma=0.99, num_episodes=500):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        done = False

        while not done:
            action_probs = policy[state]
            action = np.random.choice(env.actions, p=action_probs)
            next_state, reward, done = env.step(state, action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if s not in visited:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)

    return V


def mc_control(env, num_iterations, episodes_per_iteration, gamma=0.99, epsilon=0.1):
    # Initialize a random policy
    policy = {}
    states, actions = env.get_states_actions()
    for state in states:
        policy[state] = np.ones(len(actions)) / len(actions)

    Q = defaultdict(lambda: np.zeros(len(actions)))
    returns = defaultdict(list)

    for iteration in range(num_iterations):
        V = mc_prediction(env, policy, gamma, episodes_per_iteration)

        # MC Control step (policy improvement)
        for state in states:
            Q_values = np.zeros(len(actions))

            for action_idx, action in enumerate(actions):
                next_state, reward, done = env.step(state, action)
                if done:
                    Q_values[action_idx] = reward
                else:
                    Q_values[action_idx] = reward + gamma * V[next_state]

            best_action = np.argmax(Q_values)

            # Epsilon-greedy policy update
            policy[state] = np.ones(len(actions)) * epsilon / len(actions)
            policy[state][best_action] += 1.0 - epsilon

            Q[state] = Q_values

        print(f"Iteration {iteration+1}/{num_iterations} completed.")

    return policy, Q


def mc_exploring_starts(env, episodes, gamma):
    Q = Types.Q()
    policy = Types.Policy()
    returns = Types.Returns()

    for _ in range(episodes):
        state = env.reset()
        
        episode = []
        done = False
        while not done:
            action = epsilon_greedy_policy(Q, state, 1) # greedy-policy == epsilon-greedy-policy with epsilon = 1
            next_state, reward, done = env.step(state, action)

            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set() # set of (state, action)
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if not (state, action) in visited:
                returns[(state, action)].append(G)
                Q[state][action] = np.average(returns[(state, action)])
                policy[state] = argmax(Q[state])
                visited.add((state, action))

    return Q, policy


def mc_first_visit(env, episodes, epsilon, gamma):
    Q = Types.Q()
    policy = Types.Policy()
    returns = Types.Returns()

    for _ in range(episodes):
        state = env.reset()
        
        episode = []
        done = False
        while not done:
            action = epsilon_greedy_policy(Q, state, 1) # greedy-policy == epsilon-greedy-policy with epsilon = 1
            next_state, reward, done = env.step(state, action)

            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set() # set of (state, action)
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if not (state, action) in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state][action] = np.average(returns[(state, action)])

                best_action = argmax(Q[state])
                if action == best_action:
                    policy[state] = 1 - epsilon + epsilon / abs(Q[])

    return Q, policy


if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(medium_level)
    # policy, Q = mc_control(env, 10, 500, epsilon=0.2, gamma=0.99)
    # Q, policy = mc_exploring_starts(env, 10000, gamma=0.96)
    Q, policy = mc_first_visit(env, 1000, epsilon=0.1, gamma=0.95)
    # policy = Q_to_policy(Q, env)

    print(policy)

    test_policy(policy, env)
    print_policy(policy, env)
