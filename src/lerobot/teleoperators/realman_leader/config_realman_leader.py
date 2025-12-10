"""
Configuration for Realman leader teleoperator.
"""

from dataclasses import dataclass, field
from typing import List

from ..base_leader import BaseLeaderConfig, BaseLeaderEndEffectorConfig
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("realman_leader")
@dataclass
class RealmanLeaderConfig(BaseLeaderConfig):
    """
    Configuration for Realman leader.
    Params:
    - ip: str, IP address of the Realman robot controller
    - port: int, port number for the Realman robot controller
    - block: bool, if True, SDK commands will block until the action is completed
    - wait_second: float, time to wait for non-blocking commands
    - velocity: int, default velocity for joint movements (0-100)
    - joint_names: List[str], list of joint names for the robot, including gripper
    - init_type: str, initialization type, choices: 'none', 'joint', 'end_effector'
    - init_state: List[float], initial joint state for the Realman leader
    - joint_units: List[str], units for robot joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    """

    ##### Realman SDK settings #####
    # IP and port of the Realman robot controller
    ip: str = "169.254.128.18"
    port: int = 8080

    # Realman has 7 joints + gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])
    
    # Default initial state for the Realman leader
    init_type: str = "joint"
    init_state: List[float] = field(default_factory=lambda: [
        -0.84, -2.03,  1.15,  1.15,  2.71,  1.60, -2.99, 888.00,
    ])

    # Realman SDK uses degrees for joint angles and meters for positions
    joint_units: List[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


@TeleoperatorConfig.register_subclass("realman_leader_end_effector")
@dataclass
class RealmanLeaderEndEffectorConfig(RealmanLeaderConfig, BaseLeaderEndEffectorConfig):
    """
    Configuration for Realman leader with end effector.
    """

    pass