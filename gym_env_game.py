import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MDPGame(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    STAY = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self, random_x=False):
        super(MDPGame, self).__init__()
        self.random_x = random_x
        self.level = initial_level
        self.height = len(self.level)
        self.width = len(self.level[0])
        self.actions = [self.LEFT, self.STAY, self.RIGHT]

        # Gymnasium requirements
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height),
            spaces.Discrete(self.width)
        ))

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_x:
            self.state = (0, random.randint(0, self.width - 1))
        else:
            self.state = (0, self.width // 2)
        return self.state, {}

    def step(self, action):
        y, x = self.state
        if action == self.LEFT:
            x = max(0, x - 1)
        elif action == self.RIGHT:
            x = min(self.width - 1, x + 1)
        y = min(y + 1, self.height - 1)
        self.state = (y, x)

        done = self.terminal(self.state)

        if self.level[y][x] == 1:
            reward = -100
            done = True
        elif done:
            reward = 100
        else:
            reward = -1 if action != self.STAY else 0

        return self.state, reward, done, False, {}

    def terminal(self, state):
        y, x = state
        return self.level[y][x] == 1 or y == self.height - 1

    def render(self):
        for i in range(len(self.level)):
            row = ["#" if cell == 1 else "." for cell in self.level[i]]
            if self.state[0] == i:
                row[self.state[1]] = "X"
            print("".join(row))
        print()

    def close(self):
        pass

initial_level = [
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