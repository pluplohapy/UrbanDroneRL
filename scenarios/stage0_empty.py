import numpy as np
from typing import Tuple
from scenarios.base_scenario import BaseScenario
from config import load_config


class Stage0Scenario(BaseScenario):

    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.config = load_config('0')

    def generate(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._generate_start_goal()

    def reset(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        self.cleanup()

        return self.generate(physics_client)