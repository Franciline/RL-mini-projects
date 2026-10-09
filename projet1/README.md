# DDPG vs TD3 on LunarLander: layer normalization and overestimation bias analysis 

We run DDPG and TD3 on `LunarLander-v3` (continuous actions) and study: 

1. the impact of **layer normalization** on **Q-value overestimation bias**;
2. the impact of **overestimation bias** on **performance**.

Bias is measured as the critic's `Q(s, a)` minus the Monte Carlo discounted return obtained from the same `(s, a)` pair (positive = overestimation).


## Structure
 
```
.
├── agents.py        # networks (optional LayerNorm), DDPG, TD3
├── train.py         # one run: algo, layer norm, seed -> results CSV
├── bias.py          # Q vs Monte Carlo return
├── search.py        # hyperparameter search (Optuna)
├── plots.py         # results -> learning curves, bias curves, stats
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
source .venv/bin/activate
pip install -e .
```

## Usage


To train a DDPG with no layer norm, 8K steps:
```bash
cd projet1
python train.py --algo ddpg --layer-norm 0 --steps 8000 --eval-every 1000 --n-eval 3
```

Or for td3 with layer norm:
```bash
python train.py --algo td3 --layer-norm 1 --steps 8000 --eval-every 1000 --n-eval 3
```

Run five independent training runs, using seeds 0 through 4:

```bash
python train.py --algo td3 --layer-norm 1 --n-runs 5 --steps 100000
```

Run DDPG and TD3, each with and without LayerNorm, through one shared worker pool:

```bash
python train.py --all-variants --n-runs 10 --jobs 5 --steps 500000
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
| `--batch-size` | positive integer | `100` | Replay-buffer transitions sampled per gradient update. |
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

The selected DDPG configuration uses `actor_lr=1.91e-5`,
`critic_lr=1.96e-4`, `gamma=0.9847`, `tau=0.0110`, `sigma=0.2921`, and
`batch_size=128`. The selected TD3 configuration uses `actor_lr=6.94e-4`,
`critic_lr=7.20e-4`, `gamma=0.9937`, `tau=0.0188`, `sigma=0.2363`,
`policy_delay=3`, `target_noise=0.2776`, `target_noise_clip=0.5296`, and
`batch_size=256`. Both use `[256, 256]` hidden layers and 5,000 random warm-up
steps. These configurations are used for the final long, multi-seed LayerNorm
comparison.

### Plots

Pass the shared timestamp produced by `--all-variants`. Plot filenames describe
their contents, for example `return.png` and `bias_mean.png`.

```bash
python plots.py 20261006-142455
```

Explicit experiment directories are still accepted for older or separately run experiments.

The command saves performance, bias, bias distribution, episode length, and Q-versus-Monte-Carlo plots in `plots/`.

To visualize 3 episodes using the model .pt and layer norm 0 or 1 depending on how it was trained:
```bash
python record.py --weights results/20261004-120000_ddpg_ln0/seed0_best.pt --layer-norm 0 --episodes 3
```

### Recording parameters

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `--weights` | path to `.pt` file | required | Actor checkpoint to load. |
| `--layer-norm` | `0`, `1` | `0` | Must match the LayerNorm setting used during training. |
| `--episodes` | positive integer | `1` | Number of episodes to record. |
| `--seed` | integer | `0` | Seed for the first episode; later episodes use consecutive seeds. |
| `--out` | directory path | `videos` | Directory where video files are saved. |


uv run python train.py --all-variants --steps 500000 --eval-every 5000 --n-eval 10 --n-runs 10

----
cmd ran:

uv run python train.py \
    --algo ddpg \
    --layer-norm 0 \
    --params search_results/20261005-231810/ddpg_trial8_params.json \
    --steps 200000 \
    --n-runs 10 \
    --jobs 2 \
    --buffer-size 100000 \
    --eval-every 4000 \
    --n-eval 5

uv run python train.py \
    --algo ddpg \
    --layer-norm 1 \
    --params search_results/20261005-231810/ddpg_trial8_params.json \
    --steps 200000 \
    --n-runs 10 \
    --jobs 2 \
    --buffer-size 100000 \
    --eval-every 4000 \
    --n-eval 5

uv run python train.py \
    --algo td3 \
    --layer-norm 0 \
    --params search_results/20261005-231809/td3_best_params.json \
    --steps 200000 \
    --n-runs 10 \
    --jobs 2 \
    --buffer-size 100000 \
    --eval-every 4000 \
    --n-eval 5

uv run python train.py \
    --algo td3 \
    --layer-norm 1 \
    --params search_results/20261005-231809/td3_best_params.json \
    --steps 200000 \
    --n-runs 10 \
    --jobs 2 \
    --buffer-size 100000 \
    --eval-every 4000 \
    --n-eval 5