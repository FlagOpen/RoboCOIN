"""
Configuration for Bi-Dummy leader.
"""

from dataclasses import dataclass, field
from typing import List

from ..base_leader import BaseLeaderConfig, BaseLeaderEndEffectorConfig
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_dummy_leader")
@dataclass
class BiDummyLeaderConfig(BaseLeaderConfig):
    """
    Configuration for Bi-Dummy leader.
    Params:
    - joint_state_left: List[float], initial joint state for left dummy leader
    - joint_state_right: List[float], initial joint state for right dummy leader
    - ee_state_left: List[float], initial end effector state for left dummy leader
    - ee_state_right: List[float], initial end effector state for right dummy leader
    """

    # default joint state for left dummy leader, 0.0 for all joints
    joint_state_left: List[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    joint_state_right: List[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    # default end effector state for left dummy leader, 0.0 for all values
    ee_state_left: List[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    ee_state_right: List[float] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])


@TeleoperatorConfig.register_subclass("bi_dummy_leader_end_effector")
@dataclass
class BiDummyLeaderEndEffectorConfig(BiDummyLeaderConfig, BaseLeaderEndEffectorConfig):
    """
    Configuration for Bi-Dummy leader with end effectors.
    """
    
    pass