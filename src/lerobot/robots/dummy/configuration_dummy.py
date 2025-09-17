from dataclasses import dataclass, field
from typing import Literal

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyRobotConfig(BaseRobotConfig):
    pass


@RobotConfig.register_subclass("dummy_end_effector")
@dataclass
class DummyRobotEndEffectorConfig(DummyRobotConfig, BaseRobotEndEffectorConfig):
    pass