from game import MDPGame
from collections import defaultdict
from utils import *

def sarsa(env: MDPGame, episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: defaultdict(lambda: 0)) # Q(s, a)

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
    Q = sarsa(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)
    V = reduce_Q_to_V(Q, env)

    print_V(V, env)
    test_policy(V, env)