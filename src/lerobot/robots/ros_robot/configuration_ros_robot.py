import numpy as np
from dataclasses import dataclass, field
from typing import Any, Literal

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from sensor_msgs.msg import JointState


@RobotConfig.register_subclass("ros_robot")
@dataclass
class ROSRobotConfig(RobotConfig):
    """
    Configuration for the ROS robot.

    Attributes:
        port: Can port for the ROS robot.
        cameras: Dictionary of camera configurations.
        init_ee_state: Initial end-effector state.
    """
    joint_names: list[str] = field(default_factory=lambda: [
        f'joint{i + 1}' for i in range(7)
    ])
    subscribers: dict[str, dict[str, Any]] = field(default_factory=dict)
    publishers: dict[str, dict[str, Any]] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # choice: joint, end_effector
    init_state_type: Literal['joint', 'end_effector'] = 'joint'
    init_state: list[int] = field(default_factory=lambda: [
        0.0, 0.4, -0.6, 0.0, 0.9, 0.0, 0.0
    ])

    def __post_init__(self):
        for each in list(self.subscribers.values()) + list(self.publishers.values()):
            if each['data_class'] == 'JointState':
                each['data_class'] = JointState
            else:
                raise ValueError(f"Unsupported data_class: {each['data_class']}")


@RobotConfig.register_subclass("ros_robot_end_effector")
@dataclass
class ROSRobotEndEffectorConfig(ROSRobotConfig):
    """
    Configuration for the ROS robot's end-effector.

    Attributes:
        port: Can port for the ROS robot.
        cameras: Dictionary of camera configurations.
        init_ee_state: Initial end-effector state.
        control_mode: Control mode for the robot, choices include 'ee_absolute', 'ee_delta_base', 'ee_delta_gripper'.
        delta_with_previous: Compute delta with previous state (True) or with initial state (False).
        base_euler: The base delta orientation from the world frame to the robot gripper frame.
        visualize: Whether to visualize the robot's observations and actions.
    """
    control_mode: str = 'ee_absolute'
    delta_with_previous: bool = True
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.5 * np.pi, 0.0])
    visualize: bool = True
