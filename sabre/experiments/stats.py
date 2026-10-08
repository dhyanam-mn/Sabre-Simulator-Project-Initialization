"""Statistical evaluation routines for SABRE experiments.

Follows Workdoc Section 8.3 and Research Report Fix F6:
- 95% Confidence Intervals via Student's t-distribution
- 95% Bootstrap Confidence Intervals (percentile method)
- Paired Wilcoxon signed-rank test
- Cliff's delta non-parametric effect size
- Holm-Bonferroni family-wise error rate multiple-testing correction
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import scipy.stats


def mean_ci_t(
    data: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute sample mean and 95% CI using Student's t-distribution.

    Returns: (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean

    sem = float(scipy.stats.sem(arr))
    if sem == 0.0:
        return mean, mean, mean

    h = float(sem * scipy.stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    return mean, mean - h, mean + h


def mean_ci_bootstrap(
    data: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute sample mean and 95% bootstrap percentile confidence interval.

    Deterministic given seed.
    Returns: (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean

    rng = np.random.Generator(np.random.PCG64(seed))
    # Resample with replacement
    indices = rng.integers(0, n, size=(n_resamples, n))
    resamples = arr[indices]
    boot_means = np.mean(resamples, axis=1)

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    return mean, ci_lower, ci_upper


def paired_wilcoxon(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Perform a paired Wilcoxon signed-rank test.

    Returns: (statistic, p_value).
    Handles zero-differences gracefully.
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)

    if len(arr_x) != len(arr_y) or len(arr_x) == 0:
        return 0.0, 1.0

    diffs = arr_x - arr_y
    nonzero = diffs[diffs != 0.0]
    if len(nonzero) == 0:
        # All pairs are identical
        return 0.0, 1.0

    try:
        res = scipy.stats.wilcoxon(arr_x, arr_y, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def cliffs_delta(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """Compute Cliff's delta non-parametric effect size.

    delta = (#(x_i > y_j) - #(x_i < y_j)) / (n_x * n_y)
    Range: [-1.0, 1.0]
    Magnitudes: |delta| < 0.147 (negligible), < 0.33 (small), < 0.474 (medium), >= 0.474 (large)
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    nx = len(arr_x)
    ny = len(arr_y)
    if nx == 0 or ny == 0:
        return 0.0

    # Efficient pairwise comparison via broadcasting
    diff_matrix = arr_x[:, np.newaxis] - arr_y[np.newaxis, :]
    greater = np.sum(diff_matrix > 0)
    lesser = np.sum(diff_matrix < 0)

    delta = float((greater - lesser) / (nx * ny))
    return delta


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down correction for multiple comparisons.

    Args:
        p_values: Sequence of unadjusted p-values.
        alpha: Family-wise error rate threshold (default 0.05).

    Returns:
        List of (adjusted_p_value, is_significant) in the same order as input p_values.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value ascending
    indexed_p = sorted(enumerate(p_values), key=lambda item: item[1])
    
    # Step-down adjusted p-values
    # adj_p_{(i)} = min(1.0, max_{k <= i} (m - k + 1) * p_{(k)})
    adj_p_sorted: list[float] = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj = min(1.0, multiplier * p_val)
        running_max = max(running_max, adj)
        adj_p_sorted[rank] = min(1.0, running_max)

    # Reconstruct original order
    results: list[tuple[float, bool]] = [None] * m  # type: ignore
    for rank, (orig_idx, _) in enumerate(indexed_p):
        adj_p = adj_p_sorted[rank]
        results[orig_idx] = (adj_p, adj_p <= alpha)

    return results
