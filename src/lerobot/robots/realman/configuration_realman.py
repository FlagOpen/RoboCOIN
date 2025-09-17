from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("realman")
@dataclass
class RealmanConfig(BaseRobotConfig):
    ip: str = "169.254.128.18"
    port: int = 8080
    block: bool = False
    wait_second: float = 0.1
    velocity: int = 30

    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    init_type: str = "joint"
    init_state: list[float] = field(default_factory=lambda: [
        -0.84, -2.03,  1.15,  1.15,  2.71,  1.60, -2.99, 888.00,
    ])

    joint_units: list[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


@RobotConfig.register_subclass("realman_end_effector")
@dataclass
class RealmanEndEffectorConfig(RealmanConfig, BaseRobotEndEffectorConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])