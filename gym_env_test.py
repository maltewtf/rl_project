import gymnasium as gym
import gym_env_register
import numpy as np
if __name__ == "__main__":
    env = gym.make('MDPGame-v0')
    obs, info = env.reset()

    print("Initial state:", obs)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        print(f"Action: {action}, State: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break

    env.close()