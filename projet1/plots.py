import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")


REQUIRED_COLUMNS = {
    "step",
    "return",
    "length",
    "bias_mean",
    "bias_mae",
    "q_mean",
    "g_mean",
}
COMPARISON_KEYS = (
    "steps",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "eval_every",
    "n_eval",
)
LAYER_LABELS = {0: "without LayerNorm", 1: "with LayerNorm"}
LAYER_COLORS = {0: "tab:blue", 1: "tab:orange"}


def load_experiments(paths):
    frames, bias_arrays, configs = [], {}, []
    for path in map(Path, paths):
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing {config_path}")
        with config_path.open() as file:
            config = json.load(file)
        configs.append(config)

        csv_paths = sorted(path.glob("seed*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"no seed CSV files in {path}")
        for csv_path in csv_paths:
            match = re.fullmatch(r"seed(-?\d+)\.csv", csv_path.name)
            if not match:
                continue
            seed = int(match.group(1))
            frame = pd.read_csv(csv_path)
            missing = REQUIRED_COLUMNS - set(frame.columns)
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            frame["algo"] = config["algo"]
            frame["layer_norm"] = int(config["layer_norm"])
            frame["seed"] = seed
            frames.append(frame)

            for stage in ("first", "last"):
                bias_path = path / f"seed{seed}_bias_{stage}.npy"
                if bias_path.is_file():
                    key = (config["algo"], int(config["layer_norm"]), stage)
                    bias_arrays.setdefault(key, []).append(np.load(bias_path))

    reference = configs[0]
    for config in configs[1:]:
        different = [key for key in COMPARISON_KEYS if config[key] != reference[key]]
        if different:
            raise ValueError(f"incompatible experiment settings: {different}")

    data = pd.concat(frames, ignore_index=True)
    duplicated = data.duplicated(["algo", "layer_norm", "seed", "step"])
    if duplicated.any():
        raise ValueError("duplicate algorithm/LayerNorm/seed/step results")
    return data, bias_arrays


def save_figure(fig, output_dir, timestamp, name):
    path = output_dir / f"{timestamp}_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_metric(data, metric, ylabel, title, output_dir, timestamp, zero=False):
    algorithms = sorted(data["algo"].unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(6 * len(algorithms), 4), squeeze=False)
    for ax, algorithm in zip(axes[0], algorithms):
        subset = data[data["algo"] == algorithm]
        for layer_norm in sorted(subset["layer_norm"].unique()):
            values = subset[subset["layer_norm"] == layer_norm]
            summary = values.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            std = summary["std"].fillna(0)
            color = LAYER_COLORS[layer_norm]
            ax.plot(summary["step"], summary["mean"], color=color,
                    label=LAYER_LABELS[layer_norm])
            ax.fill_between(summary["step"], summary["mean"] - std,
                            summary["mean"] + std, color=color, alpha=0.2)
        if zero:
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(algorithm.upper())
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"{title} (mean ± std across training seeds)")
    save_figure(fig, output_dir, timestamp, metric)


def plot_bias_vs_performance(data, output_dir, timestamp):
    algorithms = sorted(data["algo"].unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(6 * len(algorithms), 4), squeeze=False)
    for ax, algorithm in zip(axes[0], algorithms):
        subset = data[data["algo"] == algorithm]
        for layer_norm in sorted(subset["layer_norm"].unique()):
            values = subset[subset["layer_norm"] == layer_norm]
            x, y = values["bias_mean"].to_numpy(), values["return"].to_numpy()
            color = LAYER_COLORS[layer_norm]
            label = LAYER_LABELS[layer_norm]
            if len(x) > 1 and np.ptp(x) > 0:
                correlation = np.corrcoef(x, y)[0, 1]
                label += f" (r={correlation:.2f})"
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(x.min(), x.max(), 100)
                ax.plot(line_x, slope * line_x + intercept, color=color, linewidth=1)
            ax.scatter(x, y, color=color, alpha=0.65, label=label)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(algorithm.upper())
        ax.set_xlabel("mean signed bias Q(s,a) - G")
        ax.set_ylabel("mean episode return")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Overestimation bias versus performance")
    save_figure(fig, output_dir, timestamp, "bias_vs_performance")


def plot_bias_distributions(data, bias_arrays, output_dir, timestamp):
    algorithms = sorted(data["algo"].unique())
    fig, axes = plt.subplots(len(algorithms), 2, figsize=(12, 4 * len(algorithms)), squeeze=False)
    for row, algorithm in enumerate(algorithms):
        for column, stage in enumerate(("first", "last")):
            ax = axes[row, column]
            for layer_norm in sorted(data.loc[data["algo"] == algorithm, "layer_norm"].unique()):
                arrays = bias_arrays.get((algorithm, layer_norm, stage), [])
                if arrays:
                    ax.hist(np.concatenate(arrays), bins=50, density=True, alpha=0.4,
                            color=LAYER_COLORS[layer_norm], label=LAYER_LABELS[layer_norm])
            ax.axvline(0, color="black", linewidth=1, linestyle="--")
            ax.set_title(f"{algorithm.upper()} — {stage} evaluation")
            ax.set_xlabel("bias Q(s,a) - G")
            ax.set_ylabel("density")
            ax.legend()
    fig.suptitle("Bias distributions at first and last evaluations")
    save_figure(fig, output_dir, timestamp, "bias_distribution_first_last")


def plot_q_vs_mc(data, output_dir, timestamp):
    conditions = sorted(data[["algo", "layer_norm"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4), squeeze=False)
    for ax, (algorithm, layer_norm) in zip(axes[0], conditions):
        subset = data[(data["algo"] == algorithm) & (data["layer_norm"] == layer_norm)]
        for metric, label, color in (("q_mean", "critic Q", "tab:red"),
                                     ("g_mean", "Monte Carlo G", "tab:green")):
            summary = subset.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            std = summary["std"].fillna(0)
            ax.plot(summary["step"], summary["mean"], color=color, label=label)
            ax.fill_between(summary["step"], summary["mean"] - std,
                            summary["mean"] + std, color=color, alpha=0.2)
        ax.set_title(f"{algorithm.upper()} — {LAYER_LABELS[layer_norm]}")
        ax.set_xlabel("training steps")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Critic prediction versus Monte Carlo return (mean ± std)")
    save_figure(fig, output_dir, timestamp, "q_vs_monte_carlo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", help="experiment result directories")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    data, bias_arrays = load_experiments(args.experiments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    plot_metric(data, "return", "mean episode return", "Performance over training",
                output_dir, timestamp)
    plot_metric(data, "bias_mean", "mean signed bias", "Overestimation bias over training",
                output_dir, timestamp, zero=True)
    plot_metric(data, "bias_mae", "mean absolute bias", "Critic error over training",
                output_dir, timestamp)
    plot_bias_vs_performance(data, output_dir, timestamp)
    plot_bias_distributions(data, bias_arrays, output_dir, timestamp)
    plot_metric(data, "length", "mean episode length", "Episode length over training",
                output_dir, timestamp)
    plot_q_vs_mc(data, output_dir, timestamp)


if __name__ == "__main__":
    main()
