import numpy as np
import random

class MDPGame:
    STAY = 0
    LEFT = -1
    RIGHT = 1

    def __init__(self, random_x=False):
        self.random_x = random_x
        self.level = [
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 1, 1, 0, 1),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 1, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 1, 1, 1, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 1, 1),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, 1, 0, 0, 1, 1),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 1, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 1, 1, 1, 1)
        ]
        self.width = len(self.level[0])
        self.height = len(self.level)
        self.actions = [self.LEFT, self.STAY, self.RIGHT]
        # self.state = (0, 1)  # Start at the top center
        self.reward = []
        self.reset() # initiate player position

    def reset(self, specific_x=None):
        """Resets the game to the starting position"""
        if self.random_x:
            self.state = (0, random.randint(0, self.width-1))
        elif specific_x != None:
            self.state = specific_x
        else:
            self.state = (0, self.width // 2)
        return self.state

    def step(self, action):
        """Calculates the next step according to the action and applies it to the state of the game"""
        self.state, reward, _ = self.get_next_state(self.state, action)
        return reward

    def evaluate_action(self, state, action):
        """Returns the simulated reward of a given action"""
        y, x = state
        done = y == self.height - 1

        if self.level[y][x] == 1: # Obstacle hit
            reward = -100
            done = True
        elif done:
            reward = 100
        else:
            if action == self.STAY:
                reward = -0.1
            else:
                reward = -2 # moving causes a small penalty to avoid pointless moves

        return reward, done

    def get_next_state(self, state, action):
        """Simulate transition without modifying the real state"""
        y, x = state
        new_x = max(0, min(self.width - 1, x + action))  # Ensure within bounds
        new_y = min(y + 1, self.height - 1)  # Always move down
        reward, done = self.evaluate_action((new_y, new_x), action)
        return (new_y, new_x), reward, done

    def get_states_actions(self):
        """Return all possible states and actions"""
        states = [(y, x) for y in range(self.height) for x in range(self.width)]
        return states, self.actions

    def terminal(self, state):
        """"Boolean True if at a terminal state, False otherwise"""
        y, x = state
        if self.level[y][x] == 1 or y == self.height - 1:
            return True
        return False

    def sample_episode(self, T=None, max_steps = 1000):
        """"Sample a random sequence from the MDP"""
        seq = []
        s = self.reset() # called to ensure each episode starts from the initial conditions
        t = 0 # step counter

        while t < max_steps:
            if T is None and self.terminal(s):
                break
            elif t == T:
                break

            a = random.choice(self.actions)

            s1, r, done = self.get_next_state(s, a) # next state, reward and game end boolean
            seq.append([s, a, r])

            s = s1
            t = t + 1
        return seq

    def print_state(self, agent_position):
        """Prints the current game state with the agent's position."""
        for i in range(len(self.level)):
            row = ["-" if j == 1 else " " for j in self.level[i]]
            if agent_position[0] == i:
                row[agent_position[1]] = "X"  # Mark agent's position
            print(f"|{''.join(row)}|")
        print("\n")