import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("base_robot")
@dataclass
class BaseRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    init_type: Literal['joint', 'end_effector', 'none'] = 'none'
    init_state: list[int] = field(default_factory=lambda: [
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
    
    delta_with: Literal['previous', 'initial', 'none'] = 'none'    
    visualize: bool = True


@RobotConfig.register_subclass("base_robot_end_effector")
@dataclass
class BaseRobotEndEffectorConfig(BaseRobotConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.5 * np.pi, 0.0])

    model_units: list[str] = field(default_factory=lambda: [
        'mm', 'mm', 'mm', 'degree', 'degree', 'degree', 'mm',
    ])
