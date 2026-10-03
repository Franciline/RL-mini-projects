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
cd proje    t1
uv sync 
source .venv.bin/activate
pip install -e .
```

## Usage


To train a DDPG with no layer norm, 20K steps: 
```bash
cd projet1
python train.py --algo ddpg --layer-norm 0 --steps 8000 --eval-every 1000 --n-eval 3
```

Or for td3 with layer norm:
```bash
python train.py --algo td3 --layer-norm 1 --steps 8000 --eval-every 1000 --n-eval 3
```

To visualize 3 episodes using the model .pt and layer norm 0 or 1 depending on how it was trained:
```bash
python record.py --weights results/*.pt --layer-norm 0 --episodes 3
```