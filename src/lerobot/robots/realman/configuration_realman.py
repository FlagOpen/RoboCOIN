from dataclasses import dataclass
from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("realman")
@dataclass
class RealmanConfig(BaseRobotConfig):
    ip: str
    port: int
    block: bool = False


@RobotConfig.register_subclass("realman_end_effector")
@dataclass
class RealmanEndEffectorConfig(RealmanConfig, BaseRobotEndEffectorConfig):
    pass