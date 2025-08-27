import numpy as np
from dataclasses import dataclass, field
from typing import Any, Literal

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("moveit_robot")
@dataclass
class MoveitRobotConfig(RobotConfig):
    """
    Configuration for the Moveit robot.

    Attributes:
        port: Can port for the ROS robot.
        cameras: Dictionary of camera configurations.
        init_ee_state: Initial end-effector state.
    """
    move_group: str = 'arm'
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # choice: joint, end_effector
    init_state_type: Literal['joint', 'end_effector'] = 'joint'
    init_state: list[int] = field(default_factory=lambda: [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    has_gripper: bool = True


@RobotConfig.register_subclass("moveit_robot_end_effector")
@dataclass
class MoveitRobotEndEffectorConfig(MoveitRobotConfig):
    """
    Configuration for the Moveit robot's end-effector.

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
