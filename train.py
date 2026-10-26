"""Compare neural-network widths using a fixed validation split."""
import json
import os
from pathlib import Path
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)
n_train, n_val, n_test = 1200, 300, 300
X = rng.uniform(-1, 1, (n_train + n_val + n_test, 2))
y = X[:, :1] * X[:, 1:2]
Xtr, Xv, Xt = np.split(X, [n_train, n_train + n_val])
ytr, yv, yt = np.split(y, [n_train, n_train + n_val])
out = Path("artifacts")
out.mkdir(exist_ok=True)

def train_width(width, generation):
    local = np.random.default_rng(SEED + width + 1000 * generation)
    W1 = local.normal(0, .35, (2, width))
    b1 = np.zeros((1, width))
    W2 = local.normal(0, .35, (width, 1))
    b2 = np.zeros((1, 1))
    params = [W1, b1, W2, b2]
    mom = [np.zeros_like(p) for p in params]
    vel = [np.zeros_like(p) for p in params]
    def predict(x):
        return np.tanh(x @ W1 + b1) @ W2 + b2
    def mse(x, target):
        return float(np.mean((predict(x) - target)**2))
    best = float("inf")
    best_params = None
    for step in range(1, 2001):
        h = np.tanh(Xtr @ W1 + b1)
        pred = h @ W2 + b2
        d = 2 * (pred - ytr) / len(Xtr)
        grads = [Xtr.T @ ((d @ W2.T) * (1 - h*h)),
                 ((d @ W2.T) * (1 - h*h)).sum(axis=0, keepdims=True),
                 h.T @ d, d.sum(axis=0, keepdims=True)]
        for i, (p, g) in enumerate(zip(params, grads)):
            mom[i] = .9*mom[i] + .1*g
            vel[i] = .999*vel[i] + .001*g*g
            p -= .01*(mom[i]/(1-.9**step))/(np.sqrt(vel[i]/(1-.999**step))+1e-8)
        if step % 25 == 0:
            val = mse(Xv, yv)
            if val < best:
                best = val
                best_params = [p.copy() for p in params]
    for p, saved in zip(params, best_params):
        p[:] = saved
    return best, best_params, mse(Xt, yt)

candidates = []
winner = None
for generation in range(3):
  for width in (8, 16, 32):
    val, weights, test_mse = train_width(width, generation)
    candidates.append({"generation": generation, "hidden_width": width, "validation_mse": val})
    if winner is None or val < winner["validation_mse"]:
        winner = {"generation": generation, "hidden_width": width, "validation_mse": val,
                  "weights": weights, "test_mse": test_mse}
np.savez(out/"best_weights.npz",
         **dict(zip(("W1", "b1", "W2", "b2"), winner["weights"])))
results = {"seed": SEED, "best_generation": winner["generation"], "best_validation_mse": winner["validation_mse"],
           "test_mse": winner["test_mse"],
           "best_hidden_width": winner["hidden_width"],
           "candidates": candidates}
(out/"history.json").write_text(json.dumps(candidates, indent=2) + "\n")
(out/"metrics.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
