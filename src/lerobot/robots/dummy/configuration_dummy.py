import numpy as np
from dataclasses import dataclass,field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyConfig(RobotConfig):
    """
    Configuration for the DummyRobot, which simulates a robot's behavior without actual hardware.
    
    Attributes:
        cameras: Dictionary of camera configurations.
        standardize: Whether to standardize the observations, seeing `.misc.standardization` for detail.
        control_mode: Control mode for the robot, choices include 'ee_absolute', 'ee_delta_base', 'ee_delta_gripper',
                      seeing `.misc.transforms` for detail.
        init_ee_state: Initial end-effector state.
        base_euler: The base delta orientation from the world frame to the robot gripper frame 
                    (only used in 'ee_delta_gripper' control mode).
        visualize: Whether to visualize the robot's observations and actions.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    standardize: bool = True
    control_mode: str = 'ee_absolute'
    init_ee_state: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0])
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    visualize: bool = True