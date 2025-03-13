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

def double_q_learning(env: MDPGame, episodes, alpha, gamma, epsilon):
    Q1 = defaultdict(lambda: defaultdict(lambda: 0)) # Q(s, a)
    Q2 = defaultdict(lambda: defaultdict(lambda: 0))

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = n_epsilon_greedy_policy([Q1, Q2], state, epsilon) # this is necessary because summing Q1 and Q2 every interation results in a massive slow down

            next_state, reward, done = env.get_next_state(state, action)
            if random.uniform(0, 1) < .5:
                best_action = max(Q1[next_state], key=Q1[next_state].get) if len(Q1[next_state]) > 0 else 0
                Q1[state][action] += alpha * (reward + gamma * Q1[next_state][best_action] - Q1[state][action])
            else:
                best_action = max(Q2[next_state], key=Q2[next_state].get) if len(Q2[next_state]) > 0 else 0
                Q2[state][action] += alpha * (reward + gamma * Q2[next_state][best_action] - Q2[state][action])

            state = next_state

    return sum_Q([Q1, Q2])

if __name__ == "__main__":
    env = MDPGame(random_x=True)
    # env.load_level(hard_level)

    Q = double_q_learning(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1)    
    V = reduce_Q_to_V(Q, env)

    print_V(V, env)
    test_policy(V, env)