"""
Configuration for Dummy robot.
"""

from dataclasses import dataclass

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyRobotConfig(BaseRobotConfig):
    """
    Configuration for Dummy robot.
    """

    pass


@RobotConfig.register_subclass("dummy_end_effector")
@dataclass
class DummyRobotEndEffectorConfig(DummyRobotConfig, BaseRobotEndEffectorConfig):
    """
    Configuration for Dummy robot end effector.
    """
    
    pass