"""Process risk scoring, containment levels, and lineage tracking.

Follows Workdoc Section 3 interface contracts exactly:
class RiskScorer:
    update(pid, f, cls, now)->float in [0,1]
    level(pid)->int 0..3
    inherit(child_pid, parent_pid)
    reasons(pid)->list[str]
"""

from __future__ import annotations

from sabre.intel.features import Features


class RiskScorer:
    """Evaluates process risk, assigns containment levels, and tracks lineage."""

    def update(self, pid: int, f: Features, cls: str, now: float) -> float:
        """Update risk score for process `pid` given features, class, and timestamp.

        Returns:
            Risk score in [0, 1].
        """
        raise NotImplementedError

    def level(self, pid: int) -> int:
        """Return containment level (0..3) for process `pid`."""
        raise NotImplementedError

    def inherit(self, child_pid: int, parent_pid: int) -> None:
        """Inherit risk state and lineage from parent process to child."""
        raise NotImplementedError

    def reasons(self, pid: int) -> list[str]:
        """Return list of explainability reasons/codes for process `pid` containment."""
        raise NotImplementedError
