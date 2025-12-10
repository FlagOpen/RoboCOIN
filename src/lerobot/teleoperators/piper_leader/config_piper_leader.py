"""
Configuration for Piper leader.
"""

from dataclasses import dataclass, field
from typing import List

from ..base_leader import BaseLeaderConfig, BaseLeaderEndEffectorConfig
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(BaseLeaderConfig):
    """
    Configuration for Piper leader.
    Params:
    - can: str, CAN bus interface for Piper leader
    - joint_names: List[str], list of joint names for each arm, including gripper
    - init_type: str, initialization type, choices: 'none', 'joint', 'end_effector'
    - init_state: List[float], initial joint state for Piper leader
    - joint_units: List[str], units for leader joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    - model_joint_units: List[str], units for model joints, for model input/output
    """

    ##### Piper SDK settings #####
    # CAN bus interface
    can: str = "can0"

    # Piper has 6 joints + gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper',
    ])
    
    init_type: str = "joint"  # 'none', 'joint', 'end_effector'
    init_state: List[float] = field(default_factory=lambda: [-1.561, 0.707, -0.558, -0.106, 0.829, 0.104, 0.069])

    # Piper SDK use 0.001 degree/mm as unit
    joint_units: List[str] = field(default_factory=lambda: [
        '001degree', '001degree', '001degree', '001degree', '001degree', '001degree', '001mm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        '001mm', '001mm', '001mm', '001degree', '001degree', '001degree', '001mm',
    ])
    # units for model joints, for model input/output
    model_joint_units: List[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    

@TeleoperatorConfig.register_subclass("piper_leader_end_effector")
@dataclass
class PiperLeaderEndEffectorConfig(PiperLeaderConfig, BaseLeaderEndEffectorConfig):
    """
    Configuration for Piper leader with end effector.
    """
    
    pass