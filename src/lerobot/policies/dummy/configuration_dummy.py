import numpy as np
from dataclasses import dataclass, field
from typing import List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("dummy")
@dataclass
class DummyConfig(PreTrainedConfig):
    """
    Configuration for the DummyPolicy, which is a placeholder policy that always returns a fixed action.
    """

    num_action_steps: int = 1
    action: List[int] | str = field(default_factory=lambda: [0.1, 0, 0, 0, 0, 0, 0])

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
