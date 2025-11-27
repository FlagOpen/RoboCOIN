"""
Configuration for Base Robot
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("base_robot")
@dataclass
class BaseRobotConfig(RobotConfig):
    """
    Configuration for Base Robot with joint control
    Params:
    - cameras: dict[str, CameraConfig], camera configurations
    - joint_names: List[str], list of joint names, including gripper
    - init_type: str, initialization type, choices: 'none', 'joint', 'end_effector'
    - init_state: List[float], initial joint state if init_type is 'joint',
      initial end effector state if init_type is 'end_effector'
    - joint_units: List[str], units for robot joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    - model_joint_units: List[str], units for model joints, for model input/output
    - delta_with: str, delta control mode, choices: 'none', 'previous', 'initial'
    - visualize: bool, visualization settings
    - draw_2d: bool, whether to draw 2D trajectories
    - draw_3d: bool, whether to draw 3D trajectories
    """

    # camera configurations, key is camera name, value is CameraConfig, e.g.
    # cameras: {"front": CameraConfig(width=640, height=480, fps=30)}
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # list of joint names, including gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])
    # initialization type, choices: 'none', 'joint', 'end_effector'
    init_type: str = 'none'
    # initial joint state if init_type is 'joint',
    # initial end effector state if init_type is 'end_effector'
    init_state: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])
    # units for robot joints, for sdk control
    joint_units: List[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    # units for end effector pose, for sdk control
    pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])
    # units for model joints, for model input/output
    model_joint_units: List[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    # delta control mode, choices: 'none', 'previous', 'initial'
    # 'none': absolute control
    # 'previous': delta control with respect to previous state
    # - no action chunking: action is delta from last state
    #   e.g. states: [s0, s1, s2], action: [a3]
    #        set s2 + a3 -> s3
    # - with action chunking: action is a sequence of deltas from current state
    #   e.g. states: [s0, s1, s2], actions: [a3, a4, a5]
    #        set s2 + a3 -> s3
    #        set s2 + a4 -> s4
    #        set s2 + a5 -> s5
    # 'initial': relative control with respect to initial state
    delta_with: str = 'none'    
    # visualization settings
    visualize: bool = True
    # whether to draw 2D trajectories
    draw_2d: bool = True
    # whether to draw 3D trajectories
    draw_3d: bool = True


@RobotConfig.register_subclass("base_robot_end_effector")
@dataclass
class BaseRobotEndEffectorConfig(BaseRobotConfig):
    """
    Configuration for Base Robot with end effector control
    Params:
    - base_euler: List[float], robot SDK control coordinate system rotation
      relative to the model coordinate system (not implemented yet)
    - model_pose_units: List[str], units for model end effector pose,
      for model input/output
    """

    # Robot SDK control coordinate system rotation
    # relative to the model coordinate system
    # (not implemented yet)
    base_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # units for model end effector pose, for model input/output
    model_pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])