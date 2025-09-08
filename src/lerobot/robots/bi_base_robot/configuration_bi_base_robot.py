from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_base_robot")
@dataclass
class BiBaseRobotConfig(BaseRobotConfig):
    init_state_left: list[int] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])
    init_state_right: list[int] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

@RobotConfig.register_subclass("bi_base_robot_end_effector")
@dataclass
class BiBaseRobotEndEffectorConfig(BiBaseRobotConfig, BaseRobotEndEffectorConfig):
    pass