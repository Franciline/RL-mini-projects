import gymnasium as gym
import numpy as np
import torch 

ENV = "LunarLander-v3"
TAIL = 300  # steps dropped at the end of a truncated episode (gamma^300 ~ 0.05)


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


def discounted_returns(rewards, gamma, terminated):
    """Monte Carlo return G_t of every step, plus a mask of reliable steps."""
    G = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    valid = np.ones(len(rewards), dtype=bool)
    if not terminated:  # cut by the time limit: the tail misses future reward
        valid[-TAIL:] = False
    return G, valid
 
 
def bias_eval(agent, seeds, gamma=0.99):
    """Q(s,a) - G_t over all steps of one deterministic episode per seed."""
    qs, gs, returns, lengths = [], [], [], []
    for seed in seeds:
        obs, act, rew, term = rollout(agent.actor, seed)
        G, valid = discounted_returns(rew, gamma, term)
        with torch.no_grad():
            q = agent.critic(torch.as_tensor(obs), torch.as_tensor(act)).numpy()
        returns.append(float(rew.sum()))
        lengths.append(len(rew))
        qs.append(q[valid])
        gs.append(G[valid])
    q, g = np.concatenate(qs), np.concatenate(gs)
    bias = q - g
    n = len(bias)
    metrics = {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
        "bias_mean": float(bias.mean()) if n else float("nan"),
        "bias_mae": float(np.abs(bias).mean()) if n else float("nan"),
        "bias_norm": float(bias.mean() / np.abs(g).mean()) if n else float("nan"),
        "q_mean": float(q.mean()) if n else float("nan"),
        "g_mean": float(g.mean()) if n else float("nan"),
        "n_pairs": n,
    }
    return metrics, bias
 


