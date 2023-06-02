from gym.envs.registration import register
register(
    id="EPEnv-v0",
    entry_point="energyplus_gymEnv.envs:EnergyPlusEnv_v0",
)