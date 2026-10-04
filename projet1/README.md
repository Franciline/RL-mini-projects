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

### Training parameters

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `--algo` | `ddpg`, `td3` | `ddpg` | Reinforcement-learning algorithm to train. |
| `--layer-norm` | `0`, `1` | `0` | Disable (`0`) or enable (`1`) LayerNorm in actor and critic hidden layers. |
| `--seed` | integer | `0` | Seed used for Python, NumPy, PyTorch, and the training environment. |
| `--steps` | positive integer | `200000` | Total number of environment transitions to collect. |
| `--buffer-size` | positive integer | `200000` | Maximum number of transitions stored in the replay buffer. |
| `--learning-starts` | non-negative integer | `5000` | Replay-buffer transitions collected before gradient updates begin. |
| `--batch-size` | positive integer | `256` | Replay-buffer transitions sampled per gradient update. |
| `--eval-every` | positive integer | `5000` | Number of training-environment steps between evaluations. |
| `--n-eval` | positive integer | `20` | Number of deterministic episodes, using fixed seeds, per evaluation. |

To visualize 3 episodes using the model .pt and layer norm 0 or 1 depending on how it was trained:
```bash
python record.py --weights results/ddpg_ln0_seed0_best.pt --layer-norm 0 --episodes 3
```

### Recording parameters

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `--weights` | path to `.pt` file | required | Actor checkpoint to load. |
| `--layer-norm` | `0`, `1` | `0` | Must match the LayerNorm setting used during training. |
| `--episodes` | positive integer | `1` | Number of episodes to record. |
| `--seed` | integer | `0` | Seed for the first episode; later episodes use consecutive seeds. |
| `--out` | directory path | `videos` | Directory where video files are saved. |
