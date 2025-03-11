from game import MDPGame
from collections import defaultdict
import numpy as np
import random
from utils import print_V

def sarsa(game: MDPGame, episodes, alpha, gamma, epsilon):
    # Q = np.zeros((game.height, game.width, 3))
    # Q = defaultdict(lambda: defaultdict(lambda: np.zeros(len(game.actions))))
    # Q = defaultdict(lambda: np.zeros(len(game.actions)))
    Q = defaultdict(lambda: defaultdict(lambda: 0)) 

    
    for episode in range(episodes):
        state = game.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False

        while not done:
            next_state, reward, done = game.get_next_state(state, action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            
            Q[state[0], state[1]][action] += alpha * (reward + gamma * Q[next_state[0], next_state[1]][action] - Q[state[0], state[1]][action])

            state = next_state
            action = next_action

    return Q

def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(-1, 1) # choose random action
    else:
        # return np.argmax(Q[state[0], state[1]])
        return max(Q[state], key=Q[state].get) if state in Q else 0

def convert_to_V(Q, game):
    # assuming Q is a defaultdict, otherwise we might get a key error
    V = {}
    for state in [(y, x) for y in range(game.height) for x in range(game.width)]:
        V[state] = max(Q[state], key=Q[state].get) if state in Q else 0

    return V

game = MDPGame(random_x=True)
Q = sarsa(game, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)
V = convert_to_V(Q, game)

inspect = (3, 1)
print(V[inspect], dict(Q[inspect]))
# V[2, 1] = 0
# print_policy_grid(V, game)
print_V(V, game)
print(V.values())

# print(V)
# print(V.keys())