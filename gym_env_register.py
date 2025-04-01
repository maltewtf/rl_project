from gymnasium.envs.registration import register

register(
    id='MDPGame-v0',
    entry_point='gym_env_game:MDPGame'
)