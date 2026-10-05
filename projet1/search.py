import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

from train import train


def sample_parameters(trial, algo, steps):
    learning_starts = [value for value in (1_000, 5_000, 10_000) if value < steps]
    if not learning_starts:
        learning_starts = [max(1, steps // 5)]

    agent = {
        "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True),
        "tau": trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        "sigma": trial.suggest_float("sigma", 0.05, 0.30),
        "gamma": trial.suggest_float("gamma", 0.98, 0.995),
    }
    if algo == "td3":
        agent.update({
            "policy_delay": trial.suggest_int("policy_delay", 2, 3),
            "target_noise": trial.suggest_float("target_noise", 0.10, 0.30),
            "target_noise_clip": trial.suggest_float(
                "target_noise_clip", 0.30, 0.70
            ),
        })
    training = {
        "batch_size": trial.suggest_categorical("batch_size", [64, 100, 128, 256]),
        "learning_starts": trial.suggest_categorical(
            "learning_starts", learning_starts
        ),
    }
    return agent, training


def objective(trial, algo, settings):
    agent_params, training_params = sample_parameters(trial, algo, settings["steps"])
    print(f"{algo.upper()} trial {trial.number} started", flush=True)
    args = argparse.Namespace(
        algo=algo,
        layer_norm=0,
        steps=settings["steps"],
        buffer_size=settings["buffer_size"],
        learning_starts=training_params["learning_starts"],
        batch_size=training_params["batch_size"],
        eval_every=settings["eval_every"],
        n_eval=settings["n_eval"],
    )

    scores, biases = [], []
    for seed in range(settings["n_runs"]):
        rows = train(
            args,
            seed,
            agent_kwargs=agent_params,
            eval_seed0=settings["eval_seed_start"],
            verbose=True,
            log_prefix=f"{algo.upper()} trial {trial.number}: ",
        )
        tail = rows[-3:]
        scores.append(float(np.mean([row["return"] for row in tail])))
        biases.append(float(np.mean([row["bias_mae"] for row in tail])))
        trial.report(float(np.mean(scores)), seed)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("seed_scores", scores)
    trial.set_user_attr("bias_mae", float(np.mean(biases)))
    return float(np.mean(scores))


def optimize_worker(storage, study_name, algo, settings, n_trials, sampler_seed):
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(lambda trial: objective(trial, algo, settings), n_trials=n_trials)


def run_study(algo, settings, output_dir, storage, jobs, trials, seed):
    study_name = f"{algo}_ln0"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    remaining = max(0, trials - len(study.trials))
    workers = min(jobs, remaining)
    sampler_seed = seed + len(study.trials)
    if workers == 1:
        optimize_worker(storage, study_name, algo, settings, remaining, sampler_seed)
    elif workers > 1:
        counts = [remaining // workers + (i < remaining % workers)
                  for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    optimize_worker, storage, study_name, algo, settings,
                    count, sampler_seed + i,
                )
                for i, count in enumerate(counts)
            ]
            for future in futures:
                future.result()

    study = optuna.load_study(study_name=study_name, storage=storage)
    study.trials_dataframe().to_csv(output_dir / f"{algo}_trials.csv", index=False)
    best = {
        "algo": algo,
        "layer_norm": False,
        **study.best_params,
        "hidden": [256, 256],
        "steps": settings["steps"],
        "n_runs": settings["n_runs"],
        "objective": study.best_value,
        "bias_mae": study.best_trial.user_attrs.get("bias_mae"),
    }
    with (output_dir / f"{algo}_best_params.json").open("w") as file:
        json.dump(best, file, indent=2)
    print(f"{algo.upper()} best return: {study.best_value:.2f}")
    print(output_dir / f"{algo}_best_params.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ddpg", "td3", "both"), default="both")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--n-runs", type=int, default=2,
                        help="training seeds per trial")
    parser.add_argument("--jobs", type=int, default=3,
                        help="maximum concurrent trials")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument("--eval-seed-start", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0, help="Optuna sampler seed")
    parser.add_argument("--output-dir",
                        help="existing directory to resume, or a new directory")
    args = parser.parse_args()

    for name in ("trials", "steps", "n_runs", "jobs", "buffer_size",
                 "eval_every", "n_eval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    created_at = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"search_results/{created_at}")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "steps": args.steps,
        "n_runs": args.n_runs,
        "buffer_size": args.buffer_size,
        "eval_every": args.eval_every,
        "n_eval": args.n_eval,
        "eval_seed_start": args.eval_seed_start,
    }
    config_path = output_dir / "config.json"
    if config_path.is_file():
        with config_path.open() as file:
            previous = json.load(file)
        comparable = ("algo", "steps", "n_runs", "buffer_size", "eval_every",
                      "n_eval", "eval_seed_start", "seed")
        changed = [name for name in comparable
                   if previous.get(name) != getattr(args, name)]
        if changed:
            parser.error(f"resume settings differ: {changed}")
        created_at = previous["created_at"]
    config = {**vars(args), "layer_norm": False, "created_at": created_at}
    with config_path.open("w") as file:
        json.dump(config, file, indent=2)

    storage = f"sqlite:///{(output_dir / 'optuna.db').resolve()}"
    algorithms = ("ddpg", "td3") if args.algo == "both" else (args.algo,)
    print(f"results: {output_dir}")
    for index, algo in enumerate(algorithms):
        run_study(algo, settings, output_dir, storage,
                  args.jobs, args.trials, args.seed + index * 10_000)


if __name__ == "__main__":
    main()
