"""
BiPiper robot configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_piper")
@dataclass
class BiPiperConfig(BiBaseRobotConfig):
    """
    BiPiper robot configuration.
    Params:
    - can_left: str, CAN bus interface for left Piper robot
    - can_right: str, CAN bus interface for right Piper robot
    - velocity: int, velocity of the robot joints (1-100)
    - joint_names: List[str], list of joint names for each arm, including gripper
    - init_state_left: List[float], initial joint state for left robot
    - init_state_right: List[float], initial joint state for right robot
    - joint_units: List[str], units for robot joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    """

    ##### BiPiper specific settings #####
    # CAN bus interfaces for left and right Piper robots
    can_left: str = "can0"
    can_right: str = "can1"
    # velocity of the robot joints (1-100)
    velocity: int = 30
    
    # Piper has 6 joints + gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper',
    ])
    
    # initial configurations for left and right robots
    init_state_left: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 60000,
    ])
    init_state_right: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 60000,
    ])
    
    # Piper SDK use 0.001 degree/mm as unit
    joint_units: List[str] = field(default_factory=lambda: [
        '001degree', '001degree', '001degree', '001degree', '001degree', '001degree', '001mm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        '001mm', '001mm', '001mm', '001degree', '001degree', '001degree', '001mm',
    ])


@RobotConfig.register_subclass("bi_piper_end_effector")
@dataclass
class BiPiperEndEffectorConfig(BiPiperConfig, BiBaseRobotEndEffectorConfig):
    """
    BiPiper robot configuration with end effectors.
    """

    pass