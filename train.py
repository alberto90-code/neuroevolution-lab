"""Reproducible NumPy neural-network experiment: learn y=x1*x2."""
import json
from pathlib import Path
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)
n_train, n_val, n_test = 1200, 300, 300
X = rng.uniform(-1, 1, (n_train+n_val+n_test, 2))
y = (X[:, 0:1] * X[:, 1:2])
Xtr, Xv, Xt = np.split(X, [n_train, n_train+n_val])
ytr, yv, yt = np.split(y, [n_train, n_train+n_val])
hidden = 32
W1 = rng.normal(0, .35, (2, hidden))
b1 = np.zeros((1, hidden))
W2 = rng.normal(0, .35, (hidden, 1))
b2 = np.zeros((1, 1))
params = [W1, b1, W2, b2]
mom = [np.zeros_like(p) for p in params]
vel = [np.zeros_like(p) for p in params]

def predict(x):
    h = np.tanh(x @ W1 + b1)
    return h @ W2 + b2

def mse(x, target):
    return float(np.mean((predict(x)-target)**2))

best = float("inf")
out = Path("artifacts")
out.mkdir(exist_ok=True)
for step in range(1, 3001):
    h = np.tanh(Xtr @ W1 + b1)
    pred = h @ W2 + b2
    d = 2 * (pred-ytr) / len(Xtr)
    gW2 = h.T @ d
    gb2 = d.sum(axis=0, keepdims=True)
    dh = (d @ W2.T) * (1-h*h)
    gW1 = Xtr.T @ dh
    gb1 = dh.sum(axis=0, keepdims=True)
    grads = [gW1, gb1, gW2, gb2]
    for i, (p, g) in enumerate(zip(params, grads)):
        mom[i] = .9*mom[i] + .1*g
        vel[i] = .999*vel[i] + .001*g*g
        p -= .01*(mom[i]/(1-.9**step))/(np.sqrt(vel[i]/(1-.999**step))+1e-8)
    if step % 25 == 0:
        val = mse(Xv, yv)
        if val < best:
            best = val
            np.savez(out/"best_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)

with np.load(out/"best_weights.npz") as saved:
    for p, name in zip(params, ("W1","b1","W2","b2")):
        p[:] = saved[name]
results = {"seed": SEED, "best_validation_mse": best, "test_mse": mse(Xt, yt)}
(out/"metrics.json").write_text(json.dumps(results, indent=2)+"\n")
print(json.dumps(results, indent=2))
