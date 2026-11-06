import math
import numpy as np


def accuracy(gold, predictions):
    if not gold or len(gold) != len(predictions):
        raise ValueError('Predictions must cover every snippet')
    if any(x not in {'A', 'B'} for x in predictions):
        raise ValueError('Predictions must be A/B')
    return sum(a == b for a, b in zip(gold, predictions)) / len(gold)


def paired(gold, predictions, baseline, seed=2026, samples=10000):
    accuracy(gold, predictions)
    accuracy(gold, baseline)
    a = np.array([x == y for x, y in zip(gold, predictions)])
    b = np.array([x == y for x, y in zip(gold, baseline)])
    delta = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(gold), size=(samples, len(gold)))].mean(axis=1)
    wins, losses = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = wins + losses, min(wins, losses)
    p = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.
    return dict(accuracy_delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                mcnemar_exact_p=p, wins=wins, losses=losses)
