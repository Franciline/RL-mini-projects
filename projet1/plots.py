import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues, pearsonr

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
    # "batch_size",
    "eval_every",
    "n_eval",
)
LAYER_LABELS = {0: "without LN", 1: "with LN"}
LAYER_STYLES = {0: "-", 1: "--"}
LAYER_COLORS = {0: "tab:green", 1: "tab:red"}
CONDITION_COLORS = {
    ("ddpg", 0): "tab:blue",
    ("ddpg", 1): "tab:blue",
    ("td3", 0): "tab:orange",
    ("td3", 1): "tab:orange",
}


def resolve_experiments(values):
    if len(values) == 1 and re.fullmatch(r"\d{8}-\d{6}", values[0]):
        timestamp = values[0]
        return [Path("results") / f"{timestamp}_{algo}_ln{layer_norm}"
                for algo in ("ddpg", "td3") for layer_norm in (0, 1)]
    return values


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
                    bias = np.load(bias_path)
                    key = (config["algo"], int(config["layer_norm"]), stage)
                    bias_arrays.setdefault(key, []).append(bias)
                    seed_key = (config["algo"], int(config["layer_norm"]), seed, stage)
                    bias_arrays.setdefault(seed_key, []).append(bias)

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


def save_figure(fig, output_dir, name):
    path = output_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def enlarge_text(axes):
    for ax in np.asarray(axes, dtype=object).flat:
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.tick_params(axis="both", labelsize=11)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(11)


def fisher_mean_r(data, metric):
    correlations = [pearsonr(values[metric], values["return"]).statistic
                    for _, values in data.groupby("seed")]
    correlations = np.clip(correlations, -1 + 1e-12, 1 - 1e-12)
    return np.tanh(np.arctanh(correlations).mean())


def plot_metric(data, metric, ylabel, title, output_dir, zero=False, scientific_x=False):
    algorithms = sorted(data["algo"].unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(6 * len(algorithms), 4), squeeze=False)
    for ax, algorithm in zip(axes[0], algorithms):
        subset = data[data["algo"] == algorithm]
        for layer_norm in sorted(subset["layer_norm"].unique()):
            values = subset[subset["layer_norm"] == layer_norm]
            summary = values.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            std = summary["std"].fillna(0)
            color = CONDITION_COLORS[(algorithm, layer_norm)]
            ax.plot(summary["step"], summary["mean"], color=color,
                    linestyle=LAYER_STYLES[layer_norm],
                    alpha=0.75 if layer_norm else 1.0,
                    label=LAYER_LABELS[layer_norm])
            ax.fill_between(summary["step"], summary["mean"] - std,
                            summary["mean"] + std, color=color, alpha=0.10)
        if zero:
            ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_title(algorithm.upper())
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylabel)
        if scientific_x:
            ax.set_xlabel("Training steps")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=False)
        ax.grid(alpha=0.25)
        ax.legend()
    large_text = metric == "length"
    if large_text:
        enlarge_text(axes)
    fig.suptitle(f"{title} (mean ± std)", fontsize=16 if large_text else None)
    save_figure(fig, output_dir, metric)


def plot_metric_combined(data, metric, ylabel, title, output_dir, zero=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = sorted(data[["algo", "layer_norm"]]
                        .drop_duplicates().itertuples(index=False, name=None))
    for algorithm, layer_norm in conditions:
        values = data[(data["algo"] == algorithm)
                      & (data["layer_norm"] == layer_norm)]
        summary = values.groupby("step")[metric].agg(["mean", "std"]).reset_index()
        std = summary["std"].fillna(0)
        color = CONDITION_COLORS[(algorithm, layer_norm)]
        label = f"{algorithm.upper()} - {LAYER_LABELS[layer_norm]}"
        ax.plot(summary["step"], summary["mean"], color=color,
                linestyle=LAYER_STYLES[layer_norm],
                alpha=0.75 if layer_norm else 1.0, label=label)
        ax.fill_between(summary["step"], summary["mean"] - std,
                        summary["mean"] + std, color=color, alpha=0.10)
    if zero:
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=False)
    ax.legend()
    large_text = metric in {"bias_mean", "return"}
    if large_text:
        enlarge_text([ax])
    fig.suptitle(f"{title} (mean ± std)", fontsize=16 if large_text else None)
    save_figure(fig, output_dir, f"{metric}_combined")


def plot_bias_vs_performance(data, output_dir, metric="bias_mean"):
    is_mae = metric == "bias_mae"
    algorithms = sorted(data["algo"].unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(6 * len(algorithms), 4), squeeze=False)
    for ax, algorithm in zip(axes[0], algorithms):
        subset = data[data["algo"] == algorithm]
        fits = []
        for layer_norm in sorted(subset["layer_norm"].unique()):
            values = subset[subset["layer_norm"] == layer_norm]
            x, y = values[metric].to_numpy(), values["return"].to_numpy()
            color = LAYER_COLORS[layer_norm]
            correlation = fisher_mean_r(values, metric)
            label = f"{LAYER_LABELS[layer_norm]} ($\\bar{{r}}$={correlation:.2f})"
            ax.scatter(x, y, s=18, color=color, alpha=0.65, label=label)
            if np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                fits.append((slope, intercept, color))
        if not is_mae:
            ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        x_limits = (subset[metric].min(), subset[metric].max())
        for slope, intercept, color in fits:
            ax.plot(x_limits, slope * np.asarray(x_limits) + intercept,
                    color=color, alpha=0.5)
        ax.set_title(algorithm.upper())
        ax.set_xlabel(r"MAE $|Q_\theta - G_t|$" if is_mae
                      else r"Mean $(Q_\theta - G_t)$")
        ax.set_ylabel("Average return")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9,
                  edgecolor="0.7") if not is_mae else ax.legend()
    if not is_mae:
        enlarge_text(axes)
    fig.suptitle("Critic Q-value MAE vs performance" if is_mae
                 else "Bias vs performance",
                 fontsize=None if is_mae else 16)
    save_figure(fig, output_dir,
                "bias_mae_vs_performance" if is_mae else "bias_vs_performance")


def plot_correlation_distribution(correlations, output_dir):
    conditions = sorted(correlations[["algo", "layer_norm"]]
                        .drop_duplicates().itertuples(index=False, name=None))
    groups = [correlations[(correlations["algo"] == algorithm)
                           & (correlations["layer_norm"] == layer_norm)]["r"].dropna()
              for algorithm, layer_norm in conditions]
    fig, ax = plt.subplots(figsize=(8, 5))
    boxes = ax.boxplot(groups, positions=np.arange(len(groups)), widths=0.45,
                       patch_artist=True, showfliers=False)
    for box, (_, layer_norm) in zip(boxes["boxes"], conditions):
        box.set(facecolor=LAYER_COLORS[layer_norm], alpha=0.15)
    for index, ((_, layer_norm), values) in enumerate(zip(conditions, groups)):
        offsets = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(index + offsets, values, s=28, color=LAYER_COLORS[layer_norm],
                   alpha=0.75, label="Training-seed r" if index == 0 else None)
        mean_r = np.tanh(np.arctanh(
            values.clip(-1 + 1e-12, 1 - 1e-12)).mean())
        ax.scatter(index, mean_r, s=75, color="black", marker="D", alpha=0.7,
                   label="Fisher-z mean r" if index == 0 else None, zorder=3)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xticks(np.arange(len(conditions)),
                  [f"{algorithm.upper()}\n{LAYER_LABELS[layer_norm]}"
                   for algorithm, layer_norm in conditions])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson r")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.7")
    enlarge_text([ax])
    fig.suptitle("Per-seed bias-performance correlations", fontsize=16)
    save_figure(fig, output_dir, "eval_bias_vs_performance_r_distribution")


def plot_eval_bias_vs_performance(data, output_dir, seed=0):
    correlations = []
    for (algorithm, layer_norm, run_seed), values in data.groupby(
            ["algo", "layer_norm", "seed"]):
        result = pearsonr(values["bias_mean"], values["return"])
        correlations.append({
            "algo": algorithm,
            "layer_norm": layer_norm,
            "seed": run_seed,
            "n_checkpoints": len(values),
            "r": result.statistic,
            "p": result.pvalue,
        })
    correlations = pd.DataFrame(correlations)
    correlations.to_csv(output_dir / "eval_bias_vs_performance_correlations.csv",
                        index=False)

    summaries = []
    for (algorithm, layer_norm), values in correlations.groupby(["algo", "layer_norm"]):
        rs = values["r"].clip(-1 + 1e-12, 1 - 1e-12)
        summaries.append({
            "algo": algorithm,
            "layer_norm": layer_norm,
            "n_seeds": len(values),
            "mean_r": values["r"].mean(),
            "mean_p": values["p"].mean(),
            "fisher_z_mean_r": np.tanh(np.arctanh(rs).mean()),
            "fisher_combined_p": combine_pvalues(values["p"], method="fisher").pvalue,
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "eval_bias_vs_performance_summary.csv", index=False)
    print(summary.to_string(index=False))
    plot_correlation_distribution(correlations, output_dir)

    algorithms = sorted(data["algo"].unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(6 * len(algorithms), 4),
                             squeeze=False)
    for ax, algorithm in zip(axes[0], algorithms):
        subset = data[(data["algo"] == algorithm) & (data["seed"] == seed)]
        for layer_norm in sorted(subset["layer_norm"].unique()):
            values = subset[subset["layer_norm"] == layer_norm]
            x, y = values["bias_mean"].to_numpy(), values["return"].to_numpy()
            result = pearsonr(x, y)
            color = LAYER_COLORS[layer_norm]
            label = (f"{LAYER_LABELS[layer_norm]} "
                     f"(r={result.statistic:.2f}, p={result.pvalue:.2g})")
            ax.scatter(x, y, s=18, color=color, alpha=0.65, label=label)
            if np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(x.min(), x.max(), 100)
                ax.plot(line_x, slope * line_x + intercept, color=color)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_title(algorithm.upper())
        ax.set_xlabel(r"Mean $(Q_\theta - G_t)$")
        ax.set_ylabel("Average return")
        ax.legend()
    fig.suptitle(f"Evaluation bias vs performance (training seed {seed})")
    save_figure(fig, output_dir, f"eval_bias_vs_performance_seed{seed}")


def plot_bias_distributions(data, bias_arrays, output_dir, seed=None):
    algorithms = sorted(data["algo"].unique())
    width = 14 if seed is None else 12
    fig, axes = plt.subplots(len(algorithms), 2, figsize=(width, 4 * len(algorithms)),
                             squeeze=False, sharey="row")
    for row, algorithm in enumerate(algorithms):
        for column, stage in enumerate(("first", "last")):
            ax = axes[row, column]
            distributions = []
            for layer_norm in sorted(data.loc[data["algo"] == algorithm, "layer_norm"].unique()):
                key = ((algorithm, layer_norm, stage) if seed is None
                       else (algorithm, layer_norm, seed, stage))
                arrays = bias_arrays.get(key, [])
                if arrays:
                    distributions.append((layer_norm, np.concatenate(arrays)))
            if distributions:
                bins = np.histogram_bin_edges(
                    np.concatenate([values for _, values in distributions]), bins=50)
                for layer_norm, values in distributions:
                    ax.hist(values, bins=bins, density=True,
                            color=LAYER_COLORS[layer_norm], alpha=0.4,
                            label=LAYER_LABELS[layer_norm])
            ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_title(f"{algorithm.upper()} - {stage} evaluation")
            ax.set_xlabel(r"Bias $Q_\theta(s,a) - G_t$")
            ax.set_ylabel("Probability density")
            if row == 0 and column == 0:
                ax.legend()
    scope = "all training seeds" if seed is None else f"training seed {seed}"
    if seed is None:
        enlarge_text(axes)
    fig.suptitle(f"Bias distributions at first and last evaluations ({scope})",
                 fontsize=16 if seed is None else None)
    name = ("bias_distribution_first_last" if seed is None
            else f"bias_distribution_seed{seed}_first_last")
    save_figure(fig, output_dir, name)


def plot_q_vs_mc(data, output_dir):
    conditions = sorted(data[["algo", "layer_norm"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4), squeeze=False)
    for index, (ax, (algorithm, layer_norm)) in enumerate(zip(axes[0], conditions)):
        subset = data[(data["algo"] == algorithm) & (data["layer_norm"] == layer_norm)]
        for metric, label, color in (("q_mean", "critic Q", "tab:purple"),
                                     ("g_mean", "Monte Carlo G", "tab:cyan")):
            summary = subset.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            std = summary["std"].fillna(0)
            ax.plot(summary["step"], summary["mean"], color=color, label=label)
            ax.fill_between(summary["step"], summary["mean"] - std,
                            summary["mean"] + std, color=color, alpha=0.10)
        ax.set_title(f"{algorithm.upper()} - {LAYER_LABELS[layer_norm]}")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Value")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=False)
        ax.grid(alpha=0.25)
        if index == 0:
            ax.legend()
    fig.suptitle("Critic prediction vs Monte Carlo return (mean ± std)")
    save_figure(fig, output_dir, "q_vs_monte_carlo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+",
                        help="shared timestamp or explicit experiment directories")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    data, bias_arrays = load_experiments(resolve_experiments(args.experiments))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(data, "return", "Mean episode return", "Performance over training",
                output_dir, scientific_x=True)
    plot_metric(data, "bias_mean", r"Mean $(Q_\theta - G_t)$",
                "Bias over training", output_dir,
                zero=True, scientific_x=True)
    plot_metric(data, "bias_mae", r"MAE $|Q_\theta - G_t|$",
                "Critic Q-value MAE", output_dir, scientific_x=True)
    plot_bias_vs_performance(data, output_dir)
    plot_bias_vs_performance(data, output_dir, metric="bias_mae")
    plot_eval_bias_vs_performance(data, output_dir, seed=0)
    plot_bias_distributions(data, bias_arrays, output_dir)
    plot_bias_distributions(data, bias_arrays, output_dir, seed=0)
    plot_metric(data, "length", "Mean episode length", "Episode length over training",
                output_dir, scientific_x=True)
    plot_q_vs_mc(data, output_dir)
    plot_metric_combined(data, "return", "Mean episode return",
                         "Performance over training", output_dir)
    plot_metric_combined(data, "bias_mean", r"Mean $(Q_\theta - G_t)$",
                         "Bias over training", output_dir, zero=True)
    plot_metric_combined(data, "bias_mae", r"MAE $|Q_\theta - G_t|$",
                         "Critic Q-value MAE", output_dir)
    plot_metric_combined(data, "length", "Mean episode length",
                         "Episode length over training", output_dir)


if __name__ == "__main__":
    main()
