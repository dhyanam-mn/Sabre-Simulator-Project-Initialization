"""Process behaviour classification interfaces and implementations.

Follows Workdoc Section 3 interface contracts exactly:
class BehaviourClassifier: classify(f: Features)->str (4 classes)
"""

from __future__ import annotations

from sabre.intel.features import Features


class BehaviourClassifier:
    """Classifies process behaviour into 4 canonical classes:
    'interactive', 'io_bound', 'cpu_bound', 'bursty'.
    """

    def classify(self, f: Features) -> str:
        """Classify a process given its extracted features."""
        raise NotImplementedError
