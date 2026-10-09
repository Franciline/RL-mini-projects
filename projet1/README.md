# DDPG vs TD3 on LunarLander: layer normalization and critic bias analysis

We run DDPG and TD3 on `LunarLander-v3` (continuous actions) and study: 

1. the impact of **layer normalization** on **Q-value bias**;
2. the relationship between **critic bias** and **performance**.

Bias is measured as the critic's `Q(s, a)` minus the Monte Carlo discounted
return obtained from the same `(s, a)` pair. Positive values indicate
overestimation; negative values indicate underestimation.


## Structure
 
```
.
├── agents.py        # networks (optional LayerNorm), DDPG, TD3
├── train.py         # one run: algo, layer norm, seed -> results CSV
├── bias.py          # Q vs Monte Carlo return
├── search.py        # hyperparameter search (Optuna)
├── plots.py         # results -> learning curves, bias curves, stats
├── record.py        # record a trained actor in LunarLander
├── assets/          # README media
├── notebooks/       # scratch only
├── results/         # one CSV per run (gitignored)
├── uv.lock          # uv files
├── pyproject.toml   
└── README.md
```

## Installation


```
cd projet1
uv sync
```

## Usage


To train a DDPG with no layer norm, 8K steps:
```bash
cd projet1
uv run python train.py --algo ddpg --layer-norm 0 --steps 8000 --eval-every 1000 --n-eval 3
```

Or for td3 with layer norm:
```bash
uv run python train.py --algo td3 --layer-norm 1 --steps 8000 --eval-every 1000 --n-eval 3
```

Run five independent training runs, using seeds 0 through 4:

```bash
uv run python train.py --algo td3 --layer-norm 1 --n-runs 5 --steps 100000
```

Run DDPG and TD3, each with and without LayerNorm, through one shared worker pool:

```bash
uv run python train.py --all-variants --n-runs 10 --jobs 5 --steps 500000
```

Each command creates one timestamped experiment directory:

```text
results/20261004-153012_td3_ln1/
├── config.json
├── seed0.csv
├── seed0_bias_first.npy
├── seed0_bias_last.npy
├── seed0_best.pt
├── seed0_final.pt
└── ...
```

### Training parameters

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `--algo` | `ddpg`, `td3` | `ddpg` | Reinforcement-learning algorithm to train. |
| `--layer-norm` | `0`, `1` | `0` | Disable (`0`) or enable (`1`) LayerNorm in actor and critic hidden layers. |
| `--all-variants` | flag | disabled | Run both algorithms with LayerNorm disabled and enabled. Ignores `--algo` and `--layer-norm`. |
| `--params` | JSON path | none | Load hyperparameters produced by `search.py`. Cannot be combined with `--all-variants`. |
| `--seed` | integer | `0` | Seed for a single run. Cannot be combined with `--n-runs` greater than 1. |
| `--n-runs` | positive integer | `1` | Number of independent runs. Multiple runs use seeds `0` through `n-runs - 1`. |
| `--jobs` | positive integer | `5` | Maximum concurrent runs. Limited automatically to `--n-runs`. |
| `--steps` | positive integer | `200000` | Total number of environment transitions to collect. |
| `--buffer-size` | positive integer | `100000` | Maximum number of transitions stored in the replay buffer. |
| `--learning-starts` | non-negative integer | `5000` | Replay-buffer transitions collected before gradient updates begin. |
| `--batch-size` | positive integer | `128` | Replay-buffer transitions sampled per gradient update. |
| `--eval-every` | positive integer | `10000` | Number of training-environment steps between evaluations after warm-up. |
| `--n-eval` | positive integer | `10` | Number of deterministic episodes, using fixed seeds, per evaluation. |

### Hyperparameter search

Search DDPG and TD3 without LayerNorm. Each trial trains two seeds by default and
is scored by the mean return over their last three checkpoints:

```bash
uv run python search.py --algo both --trials 30 --steps 50000 --n-runs 2 --jobs 3
```

Warm-up is fixed at `5000` steps for every trial. Evaluations occur at fixed
multiples of `--eval-every`, so trials are compared at identical training steps.

The command creates:

```text
search_results/<timestamp>/
├── config.json
├── optuna.db
├── ddpg_best_params.json
├── ddpg_trials.csv
├── td3_best_params.json
└── td3_trials.csv
```

Resume or extend a search by passing its directory and a larger total trial count:

```bash
uv run python search.py --algo both --trials 50 \
  --output-dir search_results/<timestamp>
```

Train with the selected parameters:

```bash
uv run python train.py --algo ddpg --layer-norm 0 \
  --params search_results/<timestamp>/ddpg_best_params.json --n-runs 10
```

#### Hyperparameter selection results

Optuna first compared 10 parameter sets per algorithm without LayerNorm, using
one seed, 30,000 training steps, and three evaluation episodes at steps 10k,
20k, and 30k. For each algorithm, the best-return trial and a lower-bias
alternative were then validated from scratch with two seeds, 50,000 steps, and
five evaluation episodes per checkpoint.

| Algorithm | Candidate | Last-three return | Final return | Final bias MAE | Choice |
| --- | --- | ---: | ---: | ---: | --- |
| DDPG | Optuna best return | -91.33 | -121.03 | 157.87 | — |
| DDPG | Lower-bias trial 8 | **-36.95** | **-35.73** | **65.78** | Selected |
| TD3 | Optuna best return | **-22.32** | **13.85** | 41.71 | Selected |
| TD3 | Lower-bias trial 0 | -53.93 | -16.77 | **40.77** | — |

#### Final hyperparameters

| Parameter | DDPG | TD3 |
| --- | ---: | ---: |
| Hidden layers | `[256, 256]` | `[256, 256]` |
| Actor learning rate | `1.9089e-5` | `6.9382e-4` |
| Critic learning rate | `1.9616e-4` | `7.2042e-4` |
| Discount factor (`gamma`) | `0.9847` | `0.9937` |
| Soft-update rate (`tau`) | `0.0110` | `0.0188` |
| Exploration noise (`sigma`) | `0.2921` | `0.2363` |
| Batch size | `128` | `256` |
| Random warm-up steps | `5000` | `5000` |
| Policy delay | — | `3` |
| Target-policy noise | — | `0.2776` |
| Target-noise clip | — | `0.5296` |

These parameters are shared by the with- and without-LayerNorm variants of
each algorithm in the final experiment.

### Final experiment

The final comparison uses 10 independent training seeds, 200,000 environment
steps, five fixed evaluation episodes every 4,000 steps, and a replay-buffer
capacity of 100,000 transitions.

```bash
uv run python train.py \
  --algo ddpg --layer-norm 0 \
  --params search_results/20261005-231810/ddpg_trial8_params.json \
  --steps 200000 --n-runs 10 --jobs 2 --buffer-size 100000 \
  --eval-every 4000 --n-eval 5

uv run python train.py \
  --algo ddpg --layer-norm 1 \
  --params search_results/20261005-231810/ddpg_trial8_params.json \
  --steps 200000 --n-runs 10 --jobs 2 --buffer-size 100000 \
  --eval-every 4000 --n-eval 5

uv run python train.py \
  --algo td3 --layer-norm 0 \
  --params search_results/20261005-231809/td3_best_params.json \
  --steps 200000 --n-runs 10 --jobs 2 --buffer-size 100000 \
  --eval-every 4000 --n-eval 5

uv run python train.py \
  --algo td3 --layer-norm 1 \
  --params search_results/20261005-231809/td3_best_params.json \
  --steps 200000 --n-runs 10 --jobs 2 --buffer-size 100000 \
  --eval-every 4000 --n-eval 5
```

### Plots

Pass the shared timestamp produced by `--all-variants`. Plot filenames describe
their contents, for example `return.png` and `bias_mean.png`.

```bash
uv run python plots.py 20261006-142455
```

Explicit experiment directories are still accepted for older or separately run experiments.

The command saves performance, bias, bias distribution, episode length, and Q-versus-Monte-Carlo plots in `plots/`.

### Policy comparison

The best saved with-LayerNorm DDPG and TD3 actors are evaluated on the same
LunarLander task (`seed=100000`). DDPG is shown on the left and TD3 on the
right. Their episode returns are 267.8 and 288.4, respectively.

<p align="center">
  <img src="assets/ddpg_vs_td3_ln.gif" alt="DDPG and TD3 LunarLander comparison" width="800">
</p>

To record episodes from another checkpoint:

```bash
uv run python record.py \
  --weights results/20261006-142455_ddpg_ln1/seed4_best.pt \
  --layer-norm 1 --episodes 3 --seed 100000
```

### Recording parameters

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `--weights` | path to `.pt` file | required | Actor checkpoint to load. |
| `--layer-norm` | `0`, `1` | `0` | Must match the LayerNorm setting used during training. |
| `--episodes` | positive integer | `1` | Number of episodes to record. |
| `--seed` | integer | `0` | Seed for the first episode; later episodes use consecutive seeds. |
| `--out` | directory path | `videos` | Directory where video files are saved. |

## Authors

- Amélie Chu
- Alan Tambellini
