"""
Base Robot class with joint control
"""

import numpy as np
from functools import cached_property
from typing import Any, Dict

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_base_robot import BaseRobotConfig
from .units_transform import UnitsTransform
from .visualization import get_visualizer


class BaseRobot(Robot):
    """
    Base class for robot implementations with joint control.
    Subclasses should implement hardware-specific communication methods.
    Supports:
    1. Joint & End-Effector control
    2. Visualization support
    3. Unified unit management
    4. Absolute & Delta action support
    Params:
    - config: BaseRobotConfig
    e.g.
    ```python
    from lerobot.robots.base_robot import BaseRobot, BaseRobotConfig

    config = BaseRobotConfig(
        joint_names=['joint1_pos', 'joint2_pos', 'joint3_pos', 'gripper'],
        init_type='joint',
        init_state=[0.0, 0.0, 0.0, 0.0],
        joint_units=['radian', 'radian', 'radian', 'meter'],
        pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
        model_joint_units=['radian', 'radian', 'radian', 'meter'],
        cameras={
            'front_camera': OpenCVCameraConfig(
                width=640,
                height=480,
                fps=30,
            ),
        },
    )
    robot = BaseRobot(config)
    robot.connect()
    observation = robot.get_observation()
    action = {'joint1_pos': 0.1, 'joint2_pos': 0.2, 'joint3_pos': 0.3}
    robot.send_action(action)
    robot.disconnect()
    ```
    """

    config_class = BaseRobotConfig
    name = "base_robot"

    def __init__(self, config: BaseRobotConfig) -> None:
        """Initialize the robot with configuration settings"""

        super().__init__(config)
        self._check_dependency()

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.arm = None
        self.visualizer = get_visualizer(
            list(self._cameras_ft.keys()), ['arm'], config.draw_2d, config.draw_3d) \
            if config.visualize else None

        self.joint_transform = UnitsTransform(config.joint_units)
        self.pose_transform = UnitsTransform(config.pose_units)
        self.model_joint_transform = UnitsTransform(config.model_joint_units)

        self._init_state = None
        self._current_state = None
    
    def _check_dependency(self) -> None:
        """
        Check for required dependencies and libraries.
        Should be implemented by subclasses 
        to verify necessary hardware libraries are available.
        """
        return
    
    def _connect_arm(self):
        """
        Establish connection to the robot arm hardware.
        This method must be implemented by subclasses 
        to handle hardware-specific connection logic.
        """
        raise NotImplementedError
    
    def _disconnect_arm(self):
        """
        Disconnect from the robot arm hardware.
        This method must be implemented by subclasses 
        to handle hardware-specific disconnection logic.
        """
        raise NotImplementedError
    
    def _set_joint_state(self, state: np.ndarray):
        """
        Set joint positions on hardware.
        This method must be implemented by subclasses 
        to send joint commands to the physical robot.
        Params:
        - state: Joint positions in robot-specific units
        """
        raise NotImplementedError
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get joint positions from hardware.
        This method must be implemented by subclasses 
        to retrieve joint states from the physical robot.
        Returns:
        - state: Joint positions in robot-specific units
        """
        raise NotImplementedError
    
    def _set_ee_state(self, state: np.ndarray):
        """
        Set end-effector pose on hardware.
        This method must be implemented by subclasses 
        to send end-effector commands to the physical robot.
        Params:
        - state: End-effector pose in robot-specific units
        """
        raise NotImplementedError
    
    def _get_ee_state(self) -> np.ndarray:
        """
        Get end-effector pose from hardware.
        This method must be implemented by subclasses
        to retrieve end-effector states from the physical robot.
        Returns:
        - state: End-effector pose in robot-specific units
        """
        raise NotImplementedError

    def set_joint_state(self, state: np.ndarray):
        """
        Set joint positions with automatic unit conversion.
        Converts from standard units to robot-specific units before sending to hardware.
        Params:
        - state: Joint positions in standard units
        """
        state = self.joint_transform.output_transform(state) # standard -> joint
        self._set_joint_state(state)
    
    def get_joint_state(self) -> np.ndarray:
        """
        Get joint positions with automatic unit conversion.
        Retrieves joint positions from hardware and converts from robot-specific units to standard units.
        Returns:
        - state: Joint positions in standard units
        """
        state = self._get_joint_state()
        return self.joint_transform.input_transform(state) # joint -> standard
    
    def set_ee_state(self, state: np.ndarray):
        """
        Set end-effector pose with automatic unit conversion.
        Converts from standard units to robot-specific units before sending to hardware.
        Params:
        - state: End-effector pose in standard units
        """
        state = self.pose_transform.output_transform(state) # standard -> end_effector
        self._set_ee_state(state)
    
    def get_ee_state(self) -> np.ndarray:
        """
        Get end-effector pose with automatic unit conversion.
        Retrieves end-effector pose from hardware and converts from robot-specific units to standard units.
        Returns:
        - state: End-effector pose in standard units
        """
        state = self._get_ee_state()
        return self.pose_transform.input_transform(state) # end_effector -> standard
    
    def prepare_and_send_action(self, action: np.ndarray) -> None:
        """
        Prepare action (delta vs absolute) and send to robot.
        Handles both absolute position commands and relative delta commands.
        Params:
        - action: Joint positions in standard units
        """
        if self.config.delta_with == 'previous':
            assert self._current_state is not None, "Current state is None, please run `get_observation` first."
            action += self._current_state
        elif self.config.delta_with == 'initial':
            assert self._init_state is not None, "Initial state is None, please run `connect` first."
            action += self._init_state
        self.set_joint_state(action)
    
    def visualize(self) -> None:
        """
        Visualize the robot state using the visualizer.
        Requires that the visualizer is initialized.
        1. Retrieves the current end-effector state and camera observations.
        2. Adds the images and state to the visualizer.
        3. Plots the visualizer.
        """
        state = self.get_ee_state()
        observation = self.get_observation()
        images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
        self.visualizer.add(images, [state])
        self.visualizer.plot()
    
    def connect(self) -> None:
        """
        Connect to the robot and initialize components.
        1. Connects to all cameras.
        2. Connects to the robot arm.
        3. Warms up the cameras by capturing initial frames.
        4. Sets the robot to the initial state based on configuration.
        """
        for cam in self.cameras.values():
            cam.connect()
        self._connect_arm()

        # Warmup cameras by capturing initial frames
        # This helps stabilize camera feeds and clear initial artifacts
        if self.cameras:
            for _ in range(10):
                for cam in self.cameras.values():
                    cam.async_read()

        # Set initial state
        if self.config.init_type == 'joint':
            self.set_joint_state(np.array(self.config.init_state))
        elif self.config.init_type == 'end_effector':
            self.set_ee_state(np.array(self.config.init_state))
        self._init_state = self.get_joint_state()

    def disconnect(self) -> None:
        """
        Disconnect from the robot and clean up resources.
        1. Disconnects all cameras.
        2. Disconnects from the robot arm.
        """
        for cam in self.cameras.values():
            cam.disconnect()
        self._disconnect_arm()
    
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send action to robot and return updated state.
        Handles unit conversion and action preparation before sending to hardware.
        Params:
        - action: Dictionary of joint positions in standard units
        Returns:
        - state: Dictionary of updated joint positions in standard units
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        action = np.array([action[each] for each in self._motors_ft.keys()])
        action = self.model_joint_transform.input_transform(action) # model -> standard

        action = self.prepare_and_send_action(action)

        if self.visualizer:
            self.visualize()

        state = self.get_joint_state()
        return {k: v for k, v in zip(self._motors_ft.keys(), state)}
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get observation from robot including joint states and camera images.
        Retrieves joint states and camera images, applies unit conversions,
        and returns a combined observation dictionary.
        Returns:
        - obs_dict: Dictionary containing joint positions and camera images
                    in model units
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state = self.get_joint_state()
        state_to_send = self.model_joint_transform.output_transform(state) # standard -> model
        obs_dict = {k: v for k, v in zip(self._motors_ft.keys(), state_to_send)}

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            obs_dict[cam_key] = outputs

        self._current_state = state

        return obs_dict
    
    @property
    def _motors_ft(self) -> Dict[str, Any]:
        """
        Motor joint features dictionary.
        Returns:
        - dict mapping joint names to float types
        """
        return {
            f'{each}_pos': float for each in self.config.joint_names
        }

    @property
    def _cameras_ft(self) -> Dict[str, tuple]:
        """
        Camera features dictionary.
        Returns:
        - dict mapping camera names to (height, width, 3) tuples
        """
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        """
        Observation features dictionary.
        Combines motor and camera features.
        Returns:
        - dict mapping observation names to their types/shapes
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> Dict[str, Any]:
        """
        Action features dictionary.
        Returns:
        - dict mapping joint names to float types
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot and all cameras are connected.
        Returns:
        - bool indicating connection status
        """
        return (
            all(self.camera.is_connected for self.camera in self.cameras.values())
        )
    
    def is_calibrated(self) -> bool:
        """
        Check if the robot is calibrated.
        Returns:
        - bool indicating calibration status, True by default
        """
        return True
    
    def calibrate(self) -> None:
        """
        Calibrate the robot, doing nothing by default.
        """
        pass

    def configure(self) -> None:
        """
        Configure the robot, doing nothing by default.
        """
        pass