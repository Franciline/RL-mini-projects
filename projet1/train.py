import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from rl_mind.collectors import TransitionCollector
from rl_mind.data import ReplayBuffer
from rl_mind.env import VecEnv

from agents import DDPG, TD3
from bias import bias_eval

ENV = "LunarLander-v3"
ALGOS = {"ddpg": DDPG, "td3": TD3}
EVAL_SEED0 = 100_000  # bias/eval seeds: fixed, identical for every run and variant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ddpg", choices=list(ALGOS))
    p.add_argument("--layer-norm", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--n-eval", type=int, default=20, help="episodes (fixed seeds) per checkpoint")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = VecEnv(ENV, num_envs=1, seed=args.seed, continuous=True)
    agent = ALGOS[args.algo](env.observation_dim, env.action_dim,
                             layer_norm=bool(args.layer_norm))
    collector = TransitionCollector(env, agent.actor)
    buffer = ReplayBuffer(200_000)
    eval_seeds = [EVAL_SEED0 + i for i in range(args.n_eval)]

    rows, best_return, best_state = [], -float("inf"), None
    first_bias = last_bias = None
    next_eval = args.eval_every

    while collector.steps < args.steps:
        buffer.add(collector.collect(1))
        if len(buffer) < 5_000:  # warm-up before learning
            continue
        agent.update(buffer.sample(256))

        if collector.steps >= next_eval:
            metrics, bias = bias_eval(agent, eval_seeds, gamma=agent.gamma)
            rows.append({"step": collector.steps, **metrics})
            if first_bias is None:
                first_bias = bias
            last_bias = bias
            if metrics["return"] > best_return:
                best_return = metrics["return"]
                best_state = copy.deepcopy(agent.actor.state_dict())
            print(f"step {collector.steps}: return {metrics['return']:.1f}  "
                  f"bias {metrics['bias_mean']:.2f}  mae {metrics['bias_mae']:.2f}")
            next_eval += args.eval_every

    os.makedirs("results", exist_ok=True)
    name = f"{args.algo}_ln{args.layer_norm}_seed{args.seed}"
    pd.DataFrame(rows).to_csv(f"results/{name}.csv", index=False)
    if first_bias is not None:
        np.save(f"results/{name}_bias_first.npy", first_bias)
        np.save(f"results/{name}_bias_last.npy", last_bias)
        torch.save(best_state, f"results/{name}_best.pt")
    torch.save(agent.actor.state_dict(), f"results/{name}_final.pt")


if __name__ == "__main__":
    main()