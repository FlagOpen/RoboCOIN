"""
BiRealman robot configuration.
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_realman")
@dataclass
class BiRealmanConfig(BiBaseRobotConfig):
    """
    BiRealman robot configuration.
    Params:
    - ip_left: str, IP address for the left Realman robot
    - port_left: int, Port for the left Realman robot
    - ip_right: str, IP address for the right Realman robot
    - port_right: int, Port for the right Realman robot
    - block: bool, whether SDK commands block until action is completed
    - wait_second: float, time to wait for non-blocking commands
    - velocity: int, default velocity for joint movements (0-100)
    - joint_names: List[str], list of joint names for each arm, including gripper
    - init_type: str, initialization type, choices: 'none', 'joint', 'end_effector'
    - init_state_left: List[float], initial joint state for left robot
    - init_state_right: List[float], initial joint state for right robot
    - joint_units: List[str], units for robot joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    """

    ##### Realman SDK settings #####
    # IP and port settings for the left and right Realman robots.
    ip_left: str = "169.254.128.18"
    port_left: int = 8080
    ip_right: str = "169.254.128.19"
    port_right: int = 8080
    # Blocking mode for SDK commands
    # - If True, SDK commands will block until the action is completed
    # - If False, SDK commands will return immediately and wait for the specified time
    block: bool = False
    wait_second: float = 0.1
    # Default velocity for joint movements (0-100)
    velocity: int = 30
    
    # BiRealman robot has 7 joints and a gripper for each side
    # Joint names for both left and right robots
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])
    
    # Default initial state for the BiRealman robot
    init_type: str = "joint"
    init_state_left: List[float] = field(default_factory=lambda: [
        -0.84, -2.03,  1.15,  1.15,  2.71,  1.60, -2.99, 888.00,
    ])
    init_state_right: List[float] = field(default_factory=lambda: [
         1.16,  2.01, -0.79, -0.68, -2.84, -1.61,  2.37, 832.00,
    ])

    # Realman SDK uses degrees for joint angles and meters for positions
    joint_units: List[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


@RobotConfig.register_subclass("bi_realman_end_effector")
@dataclass
class BiRealmanEndEffectorConfig(BiRealmanConfig, BiBaseRobotEndEffectorConfig):
    """
    BiRealman robot configuration with end effectors.
    """

    pass