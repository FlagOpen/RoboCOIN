from dataclasses import dataclass, field
from typing import Literal

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_realman")
@dataclass
class BiRealmanConfig(BiBaseRobotConfig):
    ip_left: str = "169.254.128.18"
    port_left: int = 8080
    ip_right: str = "169.254.128.19"
    port_right: int = 8080
    block: bool = False
    
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    init_type: str = 'none'
    init_state_left: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])
    init_state_right: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

    model_units: list[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'mm',
    ])
    joint_units: list[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'mm',
    ])
    pose_units: list[str] = field(default_factory=lambda: [
        'mm', 'mm', 'mm', 'degree', 'degree', 'degree', 'mm',
    ])
    
    delta_with: str = 'none'    
    visualize: bool = True


@RobotConfig.register_subclass("bi_realman_end_effector")
@dataclass
class BiRealmanEndEffectorConfig(BiRealmanConfig, BiBaseRobotEndEffectorConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    model_units: list[str] = field(default_factory=lambda: [
        'mm', 'mm', 'mm', 'degree', 'degree', 'degree', 'mm',
    ])