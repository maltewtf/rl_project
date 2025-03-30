from collections import defaultdict
from game import *
from utils import *

def sarsa(env: MDPGame, episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: defaultdict(lambda: 0)) # Q(s, a) where Q is a dict whith states as keys and dicts as values which store action reward pairs

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False

        while not done:
            next_state, reward, done = env.get_next_state(state, action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            # Q(S, A) += alpha * (R + gamma * Q(S', A') - Q(S, A))
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action

    return Q

if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(hard_level)

    # Q = sarsa(env, episodes=10000, alpha=0.1, gamma=0.95, epsilon=0.15) # These parameters solved hard_level
    Q = sarsa(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)
    policy = Q_to_policy(Q, env)

    print_policy(policy, env)
    test_policy(policy, env, count_partial_success=True)