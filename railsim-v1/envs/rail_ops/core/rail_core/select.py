"""Choose a hard eval set from a candidate pool, without looking at any model's results on the eval tasks.

    python -m rail_core.select --pool ../tasks/pool-eval --calib ../../results/rollouts/<run>/index.json \
        --n 45 --out ../tasks/rail-v1-eval

How:
1. Calibrate on the train split. A model is run on a few train tasks (eval/run_eval.py --split train); its rewards
   and the tasks' hardness features (`hardness.features`, model-free) are the calibration set.
2. Fit a ridge regression from standardized features to reward on that set (pure Python, leave-one-out error
   reported). Features whose coefficient sign is unstable under leave-one-out are dropped.
3. Predict every pool task's reward (the pool is built on held-out days only) and keep the `n` with the lowest
   prediction, with at most `max_per_day` per day and every tier in the pool represented (`min_per_tier`).
The eval tasks themselves are never scored by a model before selection: the choice depends only on train-split
calibration and on features any solver-built task has, so the eval stays a fair test.

Writes the pack (tasks.jsonl, manifest.json with the selection) and selection.json: the fit, its leave-one-out
error, the weights, and every pool task's prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .build import write_pack
from .pack import TaskPack

FEATURES = ["runs", "conflicts", "decisions", "holds", "short_turns", "cancels", "turn_points", "avoidable",
            "greedy_gap", "closed_min", "slow_min", "track_min", "single_closed", "mid_line", "late_sets"]


def _vec(h: dict, feats: list[str]) -> list[float]:
    out = []
    for f in feats:
        v = h.get(f, 0)
        v = float(v) if not isinstance(v, bool) else float(int(v))
        if f in ("avoidable",):
            v = math.log1p(max(0.0, v))
        out.append(v)
    return out


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def ridge(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """w (intercept first) minimizing |y - [1 X] w|^2 + lam |w[1:]|^2."""
    Z = [[1.0] + row for row in X]
    d = len(Z[0])
    A = [[sum(z[i] * z[j] for z in Z) + (lam if i == j and i > 0 else 0.0) for j in range(d)] for i in range(d)]
    b = [sum(z[i] * yy for z, yy in zip(Z, y)) for i in range(d)]
    return _solve(A, b)


def predict(w: list[float], x: list[float]) -> float:
    return w[0] + sum(a * b for a, b in zip(w[1:], x))


class Scaler:
    def __init__(self, rows: list[list[float]]):
        n = len(rows)
        self.mu = [sum(r[j] for r in rows) / n for j in range(len(rows[0]))]
        self.sd = [math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in rows) / n) or 1.0 for j in range(len(rows[0]))]

    def __call__(self, r: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(r, self.mu, self.sd)]


def fit(calib: list[tuple[dict, float]], pool_feats: list[dict], feats: list[str], lam: float):
    X_raw = [_vec(h, feats) for h, _ in calib]
    scale = Scaler(X_raw + [_vec(h, feats) for h in pool_feats])
    X = [scale(r) for r in X_raw]
    y = [r for _, r in calib]
    w = ridge(X, y, lam)
    loo, signs = [], []
    for i in range(len(X)):
        wi = ridge(X[:i] + X[i + 1:], y[:i] + y[i + 1:], lam)
        loo.append(predict(wi, X[i]) - y[i])
        signs.append([math.copysign(1, v) if abs(v) > 1e-9 else 0 for v in wi[1:]])
    stable = [all(s[j] == signs[0][j] for s in signs) for j in range(len(feats))]
    return w, scale, loo, stable


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) or 1.0
    return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--calib", nargs="+", required=True, help="index.json of runs on train tasks (with hardness)")
    ap.add_argument("--model", default=None, help="only this model's calibration episodes")
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-day", type=int, default=3)
    ap.add_argument("--min-per-tier", type=int, default=4)
    ap.add_argument("--lam", type=float, default=3.0)
    args = ap.parse_args()
    calib = []
    for path in args.calib:
        for e in json.loads(Path(path).read_text())["episodes"]:
            if e.get("split") == "train" and e.get("hardness") and (not args.model or e["model"] == args.model):
                calib.append((e["hardness"], float(e["reward"])))
    pool = TaskPack(args.pool)
    tasks = [t for t in pool.tasks if t.split == "eval"]
    feats = list(FEATURES)
    w, scale, loo, stable = fit(calib, [t.reference["hardness"] for t in tasks], feats, args.lam)
    keep = [f for f, s in zip(feats, stable) if s]
    if keep and len(keep) < len(feats):
        feats = keep
        w, scale, loo, stable = fit(calib, [t.reference["hardness"] for t in tasks], feats, args.lam)
    preds = {t.task_id: predict(w, scale(_vec(t.reference["hardness"], feats))) for t in tasks}
    y = [r for _, r in calib]
    loo_pred = [e + yy for e, yy in zip(loo, y)]
    # pick: lowest predicted reward, every tier represented, at most max_per_day per day
    order = sorted(tasks, key=lambda t: preds[t.task_id])
    chosen, per_day, per_tier = [], {}, {}
    tiers = sorted({t.difficulty for t in tasks})
    for tier in tiers:                       # the hardest few of each tier first
        for t in [x for x in order if x.difficulty == tier]:
            if per_tier.get(tier, 0) >= args.min_per_tier or len(chosen) >= args.n:
                break
            if per_day.get(t.day, 0) < args.max_per_day:
                chosen.append(t)
                per_day[t.day] = per_day.get(t.day, 0) + 1
                per_tier[tier] = per_tier.get(tier, 0) + 1
    for t in order:
        if len(chosen) >= args.n:
            break
        if t in chosen or per_day.get(t.day, 0) >= args.max_per_day:
            continue
        chosen.append(t)
        per_day[t.day] = per_day.get(t.day, 0) + 1
        per_tier[t.difficulty] = per_tier.get(t.difficulty, 0) + 1
    report = {
        "calibration_episodes": len(calib), "features": feats, "lambda": args.lam,
        "weights": dict(zip(["intercept"] + feats, [round(v, 4) for v in w])),
        "loo_mae": round(sum(abs(e) for e in loo) / max(1, len(loo)), 4),
        "loo_spearman": round(_spearman(loo_pred, y), 4) if len(y) > 2 else None,
        "calibration_mean_reward": round(sum(y) / max(1, len(y)), 4),
        "pool": len(tasks), "chosen": len(chosen), "per_tier": per_tier,
        "predicted_mean_chosen": round(sum(preds[t.task_id] for t in chosen) / max(1, len(chosen)), 4),
        "predicted_mean_pool": round(sum(preds.values()) / max(1, len(preds)), 4),
        "predictions": {k: round(v, 4) for k, v in sorted(preds.items(), key=lambda kv: kv[1])},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in chosen:
        t.reference["predicted_reward"] = round(preds[t.task_id], 4)
    picked = sorted((t.to_dict() for t in chosen), key=lambda d: (d["difficulty"], d["day"], d["start"]))
    log = [json.loads(line) for line in (Path(args.pool) / "build_log.jsonl").read_text().splitlines()] \
        if (Path(args.pool) / "build_log.jsonl").is_file() else []
    manifest = write_pack(out, picked, log, "eval", extra={"selected_from": Path(args.pool).name,
                                                          "selection": {k: report[k] for k in report
                                                                        if k != "predictions"}})
    (out / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=1))
    print(json.dumps({k: manifest[k] for k in ("tasks", "tiers", "runs")}))


if __name__ == "__main__":
    main()
