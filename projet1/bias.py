import gymnasium as gym
import numpy as np
import torch 

ENV = "LunarLander-v3"

def rollout(actor,seed):
    """Play one full episode from a given seed with current deterministic actor. Returns observations, actions, rewards, terminated."""
    env = gym.make(ENV,continuous=True)
    obs, _ = env.reset(seed=seed) # fix the terrain and start condition, same seed gives same episode (with a deterministic actor)
    observations, actions, rewards = [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = actor.act(obs_t)[0].numpy() # actor.act returns with batch dimension 
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs
    env.close()

    return (np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32), 
            terminated) # if terminated true, then episode ended in true state with future reward 0, if false then capped at 1K step