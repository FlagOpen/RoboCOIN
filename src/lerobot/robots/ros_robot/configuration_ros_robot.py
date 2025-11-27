"""
Configuration for ROS robot.
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.robots import RobotConfig

from ..base_robot import BaseRobotConfig, BaseRobotEndEffectorConfig


@RobotConfig.register_subclass("ros_robot")
@dataclass
class ROSRobotConfig(BaseRobotConfig):
    """
    Configuration for ROS robot.
    Params:
    - ip: str, IP address of the Realman robot controller
    - port: int, port number for the Realman robot controller
    - block: bool, if True, SDK commands will block until the action is completed
    - wait_second: float, time to wait for non-blocking commands
    - velocity: int, default velocity for joint movements (0-100)
    - joint_names: List[str], list of joint names for the robot, including gripper
    - init_type: str, initialization type, choices: 'none', 'joint', 'end_effector'
    - init_state: List[float], initial joint state for the Realman robot
    - joint_units: List[str], units for robot joints, for sdk control
    - pose_units: List[str], units for end effector pose, for sdk control
    """

    ##### ROS settings #####
    # ROS topic subscribers and publishers configurations
    # - subscriber (dict): {name: str, data_class: type, queue_size: int}
    # - publisher (dict): {name: str, data_class: type, queue_size: int}
    joint_subscribers: List[dict] = field(default_factory=lambda: [])
    joint_publishers: List[dict] = field(default_factory=lambda: [])

    # Assume ROS robot has 6 joints and a gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper',
    ])

    # Default initial state for the ROS robot
    init_type: str = "joint"
    init_state: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 1000,
    ])

    # Assume ROS robot uses degrees for joint angles and meters for positions
    joint_units: List[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


@RobotConfig.register_subclass("ros_robot_end_effector")
@dataclass
class ROSRobotEndEffectorConfig(ROSRobotConfig, BaseRobotEndEffectorConfig):
    """
    Configuration for ROS robot with end effector.
    """

    pass