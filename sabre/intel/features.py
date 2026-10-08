"""Feature representations and FeatureExtractor interface.

Follows Workdoc Section 3 interface contracts plus F1 additions:
burst_age_ms (float) and slices_since_io (int).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Features:
    """Statistical features extracted from process behavior."""

    mean_burst: float
    cv_burst: float
    io_ratio: float
    cpu_share: float  # share of wall time on CPU over last T_share ms
    full_quantum_frac: float  # fraction of recent slices that used the whole quantum
    fork_rate: float  # children per 100 ms
    n_descendants: int
    burst_age_ms: float  # F1: CPU run time since last IO / burst end
    slices_since_io: int  # F1: number of execution slices without IO
