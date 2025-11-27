"""
Configuration for Bi-Base Robot
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_base_robot")
@dataclass
class BiBaseRobotConfig(BaseRobotConfig):
    """
    Configuration for Bi-Base Robot with joint control
    Params:
    - init_state_left: List[float], initial joint state for left arm
    - init_state_right: List[float], initial joint state for right arm
    """

    # list of joint names for left and right arms, including grippers
    init_state_left: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])
    init_state_right: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

@RobotConfig.register_subclass("bi_base_robot_end_effector")
@dataclass
class BiBaseRobotEndEffectorConfig(BiBaseRobotConfig, BaseRobotEndEffectorConfig):
    """
    Configuration for Bi-Base Robot with end effector control
    """
    
    pass