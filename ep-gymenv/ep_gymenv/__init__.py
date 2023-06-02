from gym.envs.registration import register
register(
    id="EPEnv-v0",
    entry_point="ep_gymenv.envs:EnergyPlusEnv_v0",
)