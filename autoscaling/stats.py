"""Seed-level summary statistics. Each training seed is one independent sample."""
import numpy as np
# Two-sided 95% Student t critical values by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.16, 14: 2.145, 15: 2.131, 16: 2.12, 17: 2.11, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.06, 30: 2.042, 40: 2.021, 60: 2.0, 120: 1.98}

def t95(df):
    return T95[max(k for k in T95 if k <= df)]

def ci95(xs):
    """Mean, sample std and half-width of the 95% t interval for the mean."""
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    n = len(xs)
    std = float(np.std(xs, ddof=1)) if n > 1 else float('nan')
    half = t95(n - 1) * std / np.sqrt(n) if n > 1 else float('nan')
    return dict(mean=float(np.mean(xs)), std=std, ci_low=float(np.mean(xs) - half), ci_high=float(np.mean(xs) + half), n=n)
