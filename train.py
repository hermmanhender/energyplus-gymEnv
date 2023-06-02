#!/usr/bin/env python
# encoding: utf-8

import os
import shutil

import gymnasium as gym
import ray
import ray.rllib.agents.ppo as ppo
from energyplus_gymEnv.envs.EPEnv import EnergyPlusEnv_v0
from ray.tune.registry import register_env


def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "EPEnv-v0"
    register_env(select_env, lambda config: EnergyPlusEnv_v0(
        {
        "epw": "C:/Users/Usuario/OneDrive - docentes.frm.utn.edu.ar/01-Desarrollo del Doctorado/Artículo 1 - MA-DQN/base de datos meteorológica/Mendoza_Obs_-hour-historico1.epw",
        'output': "E:/Usuario/Cliope/Documents/Output EPEnv",
        "idf": "E:/Usuario/Cliope/Documents/GitHub/EP_RLlib/EP_IDF_Configuration/modelo_simple_mejorado_V2210.epJSON"
        },
        0.5
    ))

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()