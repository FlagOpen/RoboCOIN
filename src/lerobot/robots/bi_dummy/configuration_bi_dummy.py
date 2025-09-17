from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_dummy")
@dataclass
class BiDummyRobotConfig(BiBaseRobotConfig):
    pass


@RobotConfig.register_subclass("bi_dummy_end_effector")
@dataclass
class BiDummyRobotEndEffectorConfig(BiDummyRobotConfig, BiBaseRobotEndEffectorConfig):
    pass