"""
Configuration for Bi-Dummy robot.
"""

from dataclasses import dataclass

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_dummy")
@dataclass
class BiDummyRobotConfig(BiBaseRobotConfig):
    """
    Configuration for Bi-Dummy robot.
    """

    pass


@RobotConfig.register_subclass("bi_dummy_end_effector")
@dataclass
class BiDummyRobotEndEffectorConfig(BiDummyRobotConfig, BiBaseRobotEndEffectorConfig):
    """
    Configuration for Bi-Dummy robot end effector.
    """
    
    pass