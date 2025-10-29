"""
Configuration for Piper robot.
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(BaseRobotConfig):
    """
    Configuration for Piper robot.
    Params:
    - can: str, CAN bus interface for Piper robot
    - velocity: int, velocity of the robot joints (1-100)
    - joint_names: List[str], list of joint names for each arm, including gripper
    - init_type: str, initialization type, choices: 'none', 'joint', 'end_effector'
    - init_state: List[float], initial joint state for Piper robot
    - joint_units: List[str], units for robot joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    """

    ##### Piper SDK settings #####
    # CAN bus interface
    can: str = "can0"
    # velocity of the robot joints (1-100)
    velocity: int = 30

    # Piper has 6 joints + gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper',
    ])

    # Initial configuration
    init_type: str = "joint"
    init_state: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 60000,
    ])

    # Piper SDK use 0.001 degree/mm as unit
    joint_units: List[str] = field(default_factory=lambda: [
        '001degree', '001degree', '001degree', '001degree', '001degree', '001degree', '001mm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        '001mm', '001mm', '001mm', '001degree', '001degree', '001degree', '001mm',
    ])
    

@RobotConfig.register_subclass("piper_end_effector")
@dataclass
class PiperEndEffectorConfig(PiperConfig, BaseRobotEndEffectorConfig):
    """
    Configuration for Piper robot with end effector.
    """
    
    pass