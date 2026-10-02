import argparse
import os

import pandas as pd
import torch
from rl_mind.collectors import TransitionCollector
from rl_mind.data import ReplayBuffer
from rl_mind.env import VecEnv
from rl_mind.evaluation import Evaluator

from agents import DDPG

ENV = "LunarLander-v3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ddpg", choices=["ddpg"])
    p.add_argument("--layer-norm", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    env = VecEnv(ENV, num_envs=1, seed=args.seed, continuous=True)
    eval_env = VecEnv(ENV, num_envs=10, seed=10_000 + args.seed, continuous=True)

    agent = DDPG(env.observation_dim, env.action_dim, layer_norm=bool(args.layer_norm))
    collector = TransitionCollector(env, agent.actor)
    buffer = ReplayBuffer(200_000)
    evaluator = Evaluator(eval_env, every=5_000)

    while collector.steps < args.steps:
        buffer.add(collector.collect(1))
        if len(buffer) < 5_000:  # warm-up before learning
            continue
        agent.update(buffer.sample(256))
        evaluator.run_if_needed(collector.steps, agent.actor)

    os.makedirs("results", exist_ok=True)
    name = f"{args.algo}_ln{args.layer_norm}_seed{args.seed}"
    df = pd.DataFrame({
        "step": [r.step for r in evaluator.history],
        "return": [float(r.mean) for r in evaluator.history],
    })
    df.to_csv(f"results/{name}.csv", index=False)
    print(df.tail())

    torch.save(evaluator.best_actor.state_dict(), f"results/{name}_best.pt")
    torch.save(agent.actor.state_dict(), f"results/{name}_final.pt")


if __name__ == "__main__":
    main()