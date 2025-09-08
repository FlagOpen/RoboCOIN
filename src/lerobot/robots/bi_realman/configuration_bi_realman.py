from dataclasses import dataclass, field
from typing import Literal

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_realman")
@dataclass
class BiRealmanConfig(BiBaseRobotConfig):
    ip_left: str
    port_left: int
    ip_right: str
    port_right: int

    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    init_type: Literal['joint', 'end_effector', 'none'] = 'none'
    init_state_left: list[int] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])
    init_state_right: list[int] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

    model_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'mm',
    ])
    joint_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'mm',
    ])
    pose_units: list[str] = field(default_factory=lambda: [
        'mm', 'mm', 'mm', 'radian', 'radian', 'radian', 'mm',
    ])
    
    delta_with: Literal['previous', 'initial', 'none'] = 'none'    
    visualize: bool = True


@RobotConfig.register_subclass("bi_realman_end_effector")
@dataclass
class BiRealmanEndEffectorConfig(BiRealmanConfig, BiBaseRobotEndEffectorConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    model_units: list[str] = field(default_factory=lambda: [
        'mm', 'mm', 'mm', 'radian', 'radian', 'radian', 'mm',
    ])