from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("base_robot")
@dataclass
class BaseRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    init_type: str = 'none'
    init_state: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

    joint_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])
    model_joint_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    
    delta_with: str = 'none'    

    visualize: bool = True
    draw_2d: bool = True
    draw_3d: bool = True


@RobotConfig.register_subclass("base_robot_end_effector")
@dataclass
class BaseRobotEndEffectorConfig(BaseRobotConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    model_pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])