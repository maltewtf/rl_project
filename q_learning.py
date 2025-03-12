from collections import defaultdict
from game import *
from utils import *

def q_learning(env: MDPGame, episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: defaultdict(lambda: 0)) # Q(s, a)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.get_next_state(state, action)
            best_action = max(Q[next_state], key=Q[next_state].get) if len(Q[next_state]) > 0 else 0

            # Q(S, A) += alpha * (R + gamma * Q(S', a) - Q(S, A))
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_action] - Q[state][action])

            state = next_state

    return Q

if __name__ == "__main__":
    env = MDPGame(random_x=True)
    env.load_level(hard_level)

    Q = q_learning(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1)
    V = reduce_Q_to_V(Q, env)

    print_V(V, env)
    test_policy(V, env)