import numpy as np
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    """
    Configuration for the Piper robot (for joint control).

    Attributes:
        port: Can port for the Piper robot.
        cameras: Dictionary of camera configurations.
        init_ee_state: Initial end-effector state.
    """
    port: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    init_ee_state: list[int] = field(default_factory=lambda: [100000, 0, 300000, 0, 90000, 0, 60000])


@RobotConfig.register_subclass("piper_end_effector")
@dataclass
class PiperEndEffectorConfig(PiperConfig):
    """
    Configuration for the Piper robot's end-effector (for end-effector control).

    Attributes:
        port: Can port for the Piper robot.
        cameras: Dictionary of camera configurations.
        init_ee_state: Initial end-effector state.
        control_mode: Control mode for the robot, choices include 'ee_absolute', 'ee_delta_base', 'ee_delta_gripper',
                      seeing `.misc.transforms` for detail.
        delta_with_previous: Compute delta with previous state (True) or with initial state (False),
                             (only used in 'ee_delta_base' or 'ee_delta_gripper' control mode).
        base_euler: The base delta orientation from the world frame to the robot gripper frame 
                    (only used in 'ee_delta_gripper' control mode).
        visualize: Whether to visualize the robot's observations and actions.
    """
    
    control_mode: str = 'ee_absolute'
    delta_with_previous: bool = True
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.5 * np.pi, 0.0])
    visualize: bool = True
