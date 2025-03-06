import numpy as np
import random

class MDPGame:
    STAY = 0
    LEFT = -1
    RIGHT = 1

    def __init__(self):
        self.width = 7
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
        self.height = len(self.level)
        self.actions = [self.LEFT, self.STAY, self.RIGHT]
        self.state = (0, 1)  # Start at the top center

    def reset(self):
        self.state = (0, 1)  # Reset to the starting position
        return self.state

    def step(self, action):
        y, x = self.state
        new_x = max(0, min(self.width - 1, x + action))
        new_y = y + 1 if y < self.height - 1 else y  # Always move downward

        # Check if game ends
        done = new_y == self.height - 1  # Reached bottom
        if self.level[new_y][new_x] == 1:  # Hit an obstacle
            reward = -100
            done = True
        elif done:
            reward = 100  # Goal reward
        else:
            reward = -1  # Step penalty (encourages faster completion)

        self.state = (new_y, new_x)
        return self.state, reward, done

    def get_next_state(self, state, action):
        """Simulate transition without modifying the real state"""
        y, x = state
        new_x = max(0, min(self.width - 1, x + action))  # Ensure within bounds
        new_y = min(y + 1, self.height - 1)  # Always move down

        done = new_y == self.height - 1  # Check if we reached the last row
        if self.level[new_y][new_x] == 1:
            return (new_y, new_x), -100, True  # Obstacle hit
        elif done:
            return (new_y, new_x), 100, True  # Goal reached
        else:
            if action == self.STAY:
                reward = -0.1  # Staying in place has no penalty
            else:
                reward = -2  # Movement has a small penalty
            return (new_y, new_x), reward, False  # Regular move

    def get_states_actions(self):
        """Return all possible states and actions"""
        states = [(y, x) for y in range(self.height) for x in range(self.width)]
        return states, self.actions

    def terminal(self, state):
        """"Boolean True if at a terminal state, False otherwise"""
        y, x = self.state
        if self.level[y][x] == 1 or y == self.height - 1:
            return True
        return False

    def sample_episode(self, policy, T = None):
        """"Sample a random sequence from the MDP"""
        seq = []
        s = self.reset

        if self.task == 'continuing':
            assert (T is not None)
            for t in range (T):
                a = random.choice(self.get_states_actions())
                s1 = self.get_next_state(self.state , a)
                (s1, reward, done) = self.step(a)
                seq.append([s, a, reward])
                s = s1
        else:
            t = 0
            while True:
                if T is None and self.terminal(s):
                    break
                elif t == T:
                    break
                a = random.choice(self.get_states_actions())
                if self.terminal(s):
                    s1 = self.reset()
                    r = 0
                else:
                    (s1, reward, done) = self.step(a)
                seq.append([s, a, reward])
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