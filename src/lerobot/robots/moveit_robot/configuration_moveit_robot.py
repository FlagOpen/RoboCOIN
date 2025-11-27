"""
# Configuration for Moveit Robot
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("moveit_robot")
@dataclass
class MoveitRobotConfig(BaseRobotConfig):
    """
    Configuration for the Moveit robot.
    """

    ##### Moveit settings #####
    # Moveit robot name
    move_group: str = 'arm'
    has_gripper: bool = True

    # Initial configuration
    init_type: str = 'joint'
    init_state: list[int] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])

    # Moveit robot use degree/meter as unit
    joint_units: List[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


@RobotConfig.register_subclass("moveit_robot_end_effector")
@dataclass
class MoveitRobotEndEffectorConfig(MoveitRobotConfig, BaseRobotEndEffectorConfig):
    """
    Configuration for Moveit robot with end effector.
    """

    pass