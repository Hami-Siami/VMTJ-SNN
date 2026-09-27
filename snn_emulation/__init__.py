"""Software-level SNN emulation package."""

from .configs import get_experiment_config
from .model import SNNClassifier

__all__ = ["SNNClassifier", "get_experiment_config"]
