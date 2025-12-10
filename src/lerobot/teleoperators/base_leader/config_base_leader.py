"""
Configuration for BaseLeader teleoperator.
"""

from dataclasses import dataclass, field
from typing import List

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("base_leader")
@dataclass
class BaseLeaderConfig(TeleoperatorConfig):
    """
    Configuration for BaseLeader teleoperator.
    Params:
    - joint_names: List[str], list of joint names, including gripper
    - joint_units: List[str], units for robot joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    - model_joint_units: List[str], units for model joints, for model input/output
    """

    # list of joint names, including gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    # initialization type and state, each episode start with this state
    # 'none': no initialization
    # 'joint': initialize joint positions
    # 'end_effector': initialize end effector pose
    init_type: str = "none"  # 'none', 'joint', 'end_effector'
    init_state: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    init_threshold: float = 0.1  # threshold to consider initialized

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


@TeleoperatorConfig.register_subclass("base_leader_end_effector")
@dataclass
class BaseLeaderEndEffectorConfig(BaseLeaderConfig):
    """
    Configuration for BaseLeaderEndEffector teleoperator.
    Extends BaseLeaderConfig with end-effector specific parameters.
    Params:
    - base_euler: List[float], robot SDK control coordinate system rotation 
      relative to the model coordinate system (not implemented yet)
    - model_pose_units: List[str], units for model end effector pose for model input/output
    """

    # robot SDK control coordinate system rotation relative to the model coordinate system
    base_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # units for model end effector pose, for model input/output
    model_pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])