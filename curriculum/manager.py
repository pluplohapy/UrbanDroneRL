"""
Curriculum manager for progressive difficulty stages.
"""

from collections import deque
from scenarios.stage0_empty import Stage0Scenario
import config


class CurriculumManager:
    """Manages curriculum learning progression through stages."""

    def __init__(self):
        """Initialize curriculum manager."""
        self.stage = 0
        self.history = deque(maxlen=config.CURRICULUM_WINDOW)
        self.thresholds = config.THRESHOLDS

    def record(self, is_success: bool):
        """
        Record episode result.

        Args:
            is_success: Whether the episode was successful
        """
        self.history.append(1 if is_success else 0)

    def should_advance(self) -> bool:
        """
        Check if should advance to next stage.

        Returns:
            True if success rate meets threshold
        """
        if self.stage not in self.thresholds:
            return False

        if len(self.history) < config.CURRICULUM_WINDOW:
            return False

        success_rate = sum(self.history) / len(self.history)
        return success_rate >= self.thresholds[self.stage]

    def advance(self):
        """Advance to next stage and reset history."""
        self.stage += 1
        self.history.clear()

    def get_scenario(self, seed=None):
        """
        Get scenario for current stage.

        Args:
            seed: Random seed

        Returns:
            Scenario object
        """
        if self.stage == 0:
            return Stage0Scenario(seed=seed)
        # Stage 1 and 2 will be added later
        else:
            return Stage0Scenario(seed=seed)

    def get_stage_name(self) -> str:
        """Get current stage name."""
        stage_names = {
            0: "Stage 0 (Empty)",
            1: "Stage 1 (Static)",
            2: "Stage 2 (Dynamic)"
        }
        return stage_names.get(self.stage, f"Stage {self.stage}")
