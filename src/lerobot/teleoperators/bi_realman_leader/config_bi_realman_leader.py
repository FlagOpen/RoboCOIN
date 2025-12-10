"""
Configuration for Bi-Realman leader.
"""

from dataclasses import dataclass, field
from typing import List

from ..base_leader import BaseLeaderConfig, BaseLeaderEndEffectorConfig
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_realman_leader")
@dataclass
class BiRealmanLeaderConfig(BaseLeaderConfig):
    """
    Configuration for Bi-Realman leader.
    """
    ##### Realman SDK settings #####
    # IP and port settings for the left and right Realman robots.
    ip_left: str = "169.254.128.18"
    port_left: int = 8080
    ip_right: str = "169.254.128.19"
    port_right: int = 8080

    # BiRealman robot has 7 joints and a gripper for each side
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    # Default initial state for the BiRealman leader
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


@TeleoperatorConfig.register_subclass("bi_realman_leader_end_effector")
@dataclass
class BiRealmanLeaderEndEffectorConfig(BiRealmanLeaderConfig, BaseLeaderEndEffectorConfig):
    """
    Configuration for Bi-Realman leader with end effector.
    """

    pass