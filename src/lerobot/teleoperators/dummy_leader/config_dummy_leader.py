"""
Configuration for Dummy Leader Teleoperator.
"""

from dataclasses import dataclass, field
from typing import List

from ..base_leader import BaseLeaderConfig, BaseLeaderEndEffectorConfig
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("dummy_leader")
@dataclass
class DummyLeaderConfig(BaseLeaderConfig):
    """
    Configuration for Dummy Leader Teleoperator.
    Params:
    - joint_state: List[float], initial joint state for dummy leader
    - ee_state: List[float], initial end effector state for dummy leader
    """

    # default joint state for dummy leader, 0.0 for all joints
    joint_state: List[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    # default end effector state for dummy leader, 0.0 for all values
    ee_state: List[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])


@TeleoperatorConfig.register_subclass("dummy_leader_end_effector")
@dataclass
class DummyLeaderEndEffectorConfig(DummyLeaderConfig, BaseLeaderEndEffectorConfig):
    """
    Configuration for Dummy Leader Teleoperator with end effector.
    """
    
    pass