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


def train(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = VecEnv(ENV, num_envs=1, seed=seed, continuous=True)
    agent = ALGOS[args.algo](env.observation_dim, env.action_dim,
                             layer_norm=bool(args.layer_norm))
    collector = TransitionCollector(env, agent.actor)
    buffer = ReplayBuffer(args.buffer_size)
    eval_seeds = [EVAL_SEED0 + i for i in range(args.n_eval)]

    rows, best_return, best_state = [], -float("inf"), None
    first_bias = last_bias = None
    next_eval = args.learning_starts

    while collector.steps < args.steps:
        buffer.add(collector.collect(1))
        if len(buffer) < args.learning_starts:  # warm-up before learning
            continue
        agent.update(buffer.sample(args.batch_size))

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
            next_eval = collector.steps + args.eval_every

    os.makedirs("results", exist_ok=True)
    name = f"{args.algo}_ln{args.layer_norm}_seed{seed}"
    pd.DataFrame(rows).to_csv(f"results/{name}.csv", index=False)
    if first_bias is not None:
        np.save(f"results/{name}_bias_first.npy", first_bias)
        np.save(f"results/{name}_bias_last.npy", last_bias)
        torch.save(best_state, f"results/{name}_best.pt")
    torch.save(agent.actor.state_dict(), f"results/{name}_final.pt")
    env.gym_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ddpg", choices=list(ALGOS))
    p.add_argument("--layer-norm", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-runs", type=int, default=1,
                   help="sequential runs; seeds are 0 to n-runs - 1")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--n-eval", type=int, default=20, help="episodes (fixed seeds) per checkpoint")
    args = p.parse_args()

    if args.n_runs < 1:
        p.error("--n-runs must be at least 1")
    if args.n_runs > 1 and args.seed != 0:
        p.error("--seed cannot be combined with --n-runs greater than 1")

    seeds = range(args.n_runs) if args.n_runs > 1 else [args.seed]
    for run, seed in enumerate(seeds, start=1):
        print(f"run {run}/{args.n_runs}: seed {seed}")
        train(args, seed)


if __name__ == "__main__":
    main()
