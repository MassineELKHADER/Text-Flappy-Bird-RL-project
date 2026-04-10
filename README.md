# Text Flappy Bird: Monte Carlo vs Sarsa(lambda)

## Requirements

```
pip install gymnasium text-flappy-bird-gym tqdm matplotlib numpy
```

## Run

Run the jupyter notebook:

Figures are saved to `figures/`.

## Files

| File | Purpose |
|------|---------|
| `notebook.ipynb` | Main experiment notebook |
| `env_utils.py` | Environment factory and state encoder |
| `agents/mc_agent.py` | First-visit Monte Carlo agent |
| `agents/sarsa_lambda.py` | Sarsa(lambda) agent |
| `utils.py` | Smoothing and evaluation helpers |
| `train.py` | Standalone training script |
