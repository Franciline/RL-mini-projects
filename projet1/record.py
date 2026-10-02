import argparse
import os

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from agents import DetActor

ENV = "LunarLander-v3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to a saved actor .pt file")
    p.add_argument("--layer-norm", type=int, default=0, help="must match the trained model")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="videos")
    args = p.parse_args()

    name = os.path.splitext(os.path.basename(args.weights))[0]
    env = RecordVideo(
        gym.make(ENV, continuous=True, render_mode="rgb_array"),
        video_folder=args.out,
        episode_trigger=lambda i: True,
        name_prefix=name,
    )

    actor = DetActor(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        layer_norm=bool(args.layer_norm),
    )
    actor.load_state_dict(torch.load(args.weights, map_location="cpu"))
    actor.eval()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, total = False, 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = actor.act(obs_t)[0].numpy()  # deterministic, no noise
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        print(f"episode {ep}: return {total:.1f}")

    env.close()


if __name__ == "__main__":
    main()