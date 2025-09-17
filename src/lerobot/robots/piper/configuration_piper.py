from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(BaseRobotConfig):
    can: str = "can0"
    velocity: int = 30

    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper',
    ])

    init_type: str = "joint"
    init_state: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 60000,
    ])

    joint_units: list[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


@RobotConfig.register_subclass("piper_end_effector")
@dataclass
class PiperEndEffectorConfig(PiperConfig, BaseRobotEndEffectorConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])