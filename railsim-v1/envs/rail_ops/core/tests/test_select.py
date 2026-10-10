"""The eval selection's regression and its pick rules (no packs needed)."""

import random

from rail_core.select import FEATURES, Scaler, _spearman, fit, predict, ridge


def test_ridge_recovers_a_linear_rule():
    rng = random.Random(0)
    X = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(40)]
    y = [0.5 + 0.3 * a - 0.2 * b for a, b in X]
    w = ridge(X, y, 1e-6)
    assert abs(w[0] - 0.5) < 1e-6 and abs(w[1] - 0.3) < 1e-6 and abs(w[2] + 0.2) < 1e-6
    assert abs(predict(w, [1.0, 1.0]) - 0.6) < 1e-6


def test_fit_reports_leave_one_out_and_stability():
    rng = random.Random(1)
    calib = []
    for _ in range(16):
        h = {f: rng.randint(0, 40) for f in FEATURES}
        h["single_closed"] = rng.random() < 0.5
        calib.append((h, max(0.2, min(1.0, 0.9 - 0.01 * h["decisions"]))))
    w, scale, loo, stable = fit(calib, [h for h, _ in calib], list(FEATURES), 3.0)
    assert len(loo) == 16 and len(stable) == len(FEATURES)
    assert w[1 + FEATURES.index("decisions")] < 0          # more decisions -> lower predicted reward
    s = Scaler([[1.0, 2.0], [3.0, 2.0]])
    assert s([2.0, 2.0]) == [0.0, 0.0]
    assert _spearman([1, 2, 3], [10, 20, 30]) == 1.0
