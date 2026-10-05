import argparse
import copy
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import gymnasium as gym
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


def train(args, seed, result_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = VecEnv(ENV, num_envs=1, seed=seed, continuous=True)
    agent = ALGOS[args.algo](env.observation_dim, env.action_dim,
                             layer_norm=bool(args.layer_norm))
    collector = TransitionCollector(env, agent.actor)
    buffer = ReplayBuffer(args.buffer_size)
    eval_seeds = [EVAL_SEED0 + i for i in range(args.n_eval)]
    eval_env = gym.make_vec(ENV, num_envs=args.n_eval, continuous=True)

    rows, best_return, best_state = [], -float("inf"), None
    first_bias = last_bias = None
    next_eval = args.learning_starts

    while collector.steps < args.steps:
        buffer.add(collector.collect(1))
        if len(buffer) < args.learning_starts:  # warm-up before learning
            continue
        agent.update(buffer.sample(args.batch_size))

        if collector.steps >= next_eval:
            metrics, bias = bias_eval(agent, eval_seeds, eval_env, gamma=agent.gamma)
            rows.append({"step": collector.steps, **metrics})
            if first_bias is None:
                first_bias = bias
            last_bias = bias
            if metrics["return"] > best_return:
                best_return = metrics["return"]
                best_state = copy.deepcopy(agent.actor.state_dict())
            print(f"{args.algo} ln{args.layer_norm}, seed {seed}, "
                  f"step {collector.steps}: return {metrics['return']:.1f}  "
                  f"bias {metrics['bias_mean']:.2f}  mae {metrics['bias_mae']:.2f}")
            next_eval = collector.steps + args.eval_every

    name = f"seed{seed}"
    pd.DataFrame(rows).to_csv(os.path.join(result_dir, f"{name}.csv"), index=False)
    if first_bias is not None:
        np.save(os.path.join(result_dir, f"{name}_bias_first.npy"), first_bias)
        np.save(os.path.join(result_dir, f"{name}_bias_last.npy"), last_bias)
        torch.save(best_state, os.path.join(result_dir, f"{name}_best.pt"))
    torch.save(agent.actor.state_dict(), os.path.join(result_dir, f"{name}_final.pt"))
    eval_env.close()
    env.gym_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ddpg", choices=list(ALGOS))
    p.add_argument("--layer-norm", type=int, default=0)
    p.add_argument("--all-variants", action="store_true",
                   help="run both algorithms with and without LayerNorm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-runs", type=int, default=1,
                   help="independent runs; seeds are 0 to n-runs - 1")
    p.add_argument("--jobs", type=int, default=5,
                   help="maximum concurrent runs")
    p.add_argument("--steps", type=int, default=200_000) # not 1M
    p.add_argument("--buffer-size", type=int, default=100_000) # changed, run Alan 500k, me 100k 
    p.add_argument("--learning-starts", type=int, default=5_000) # warmup
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--n-eval", type=int, default=10, help="episodes (fixed seeds) per checkpoint")
    args = p.parse_args()

    if args.n_runs < 1:
        p.error("--n-runs must be at least 1")
    if args.jobs < 1:
        p.error("--jobs must be at least 1")
    if args.n_runs > 1 and args.seed != 0:
        p.error("--seed cannot be combined with --n-runs greater than 1")

    variants = ([(algo, layer_norm) for algo in ALGOS for layer_norm in (0, 1)]
                if args.all_variants else [(args.algo, args.layer_norm)])
    seeds = list(range(args.n_runs)) if args.n_runs > 1 else [args.seed]
    created_at = datetime.now().strftime("%Y%m%d-%H%M%S")
    tasks = []
    for algo, layer_norm in variants:
        variant_args = copy.copy(args)
        variant_args.algo, variant_args.layer_norm = algo, layer_norm
        result_dir = os.path.join("results", f"{created_at}_{algo}_ln{layer_norm}")
        os.makedirs(result_dir)
        config = {
            **vars(variant_args),
            "environment": ENV,
            "evaluation_seed_start": EVAL_SEED0,
            "created_at": created_at,
        }
        with open(os.path.join(result_dir, "config.json"), "w") as file:
            json.dump(config, file, indent=2)
        print(f"results: {result_dir}")
        tasks.extend((variant_args, seed, result_dir) for seed in seeds)

    workers = min(args.jobs, len(tasks))
    if workers == 1:
        for task in tasks:
            train(*task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(train, *task) for task in tasks]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
