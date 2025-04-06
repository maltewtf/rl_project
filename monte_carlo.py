import numpy as np
from game import *
from utils import *

def mc_epsilon_greedy(env, episodes, epsilon, gamma):
    """
    Applies Monte Carlo methods with an ε-greedy strategy.

    returns: Q
    """

    Q = Types.Q()
    returns = Types.Returns()

    for _ in range(episodes):
        state = env.reset()
        done = False

        episode = [] # Generate an episode using the current ε-greedy policy.
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(state, action)

            episode.append((state, action, reward))
            state = next_state

        G = 0  # cumulative discounted return
        visited = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward

            # Update only on the first occurrence of (s, a)
            if not (state, action) in visited:
                returns[(state, action)].append(G)
                Q[state][action] = np.average(returns[(state, action)])

                visited.add((state, action))

    return Q


if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(medium_level)
    Q = mc_epsilon_greedy(env, 2000, epsilon=0.1, gamma=0.95)
    policy = Q_to_policy(Q, env)

    test_policy(policy, env, count_partial_success=True)
    print_policy(policy, env)
