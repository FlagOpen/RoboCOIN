import numpy as np
from functools import cached_property
from typing import Any, Dict

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .configuration_bi_base_robot import BiBaseRobotConfig
from ..base_robot.visualization import get_visualizer


class BiBaseRobot(Robot):
    """
    Base class for dual-arm robot systems.
    Manages two independent robot arms (left and right) along with cameras.
    Delegates operations to individual arm controllers while providing unified interface.
    Params:
    - config: BiBaseRobotConfig
    e.g.
    ```python
    from lerobot.robots.bi_base_robot import BiBaseRobot
    from lerobot.robots.bi_base_robot.configuration_bi_base_robot import BiBaseRobotConfig

    config = BiBaseRobotConfig(...)
    robot = BiBaseRobot(config)
    robot.connect()
    obs = robot.get_observation()
    action = {...}  # Define actions for both arms
    robot.send_action(action)
    robot.disconnect()
    ```
    """

    config_class = BiBaseRobotConfig
    name = "base_robot"

    def __init__(self, config: BiBaseRobotConfig) -> None:
        """
        Initialize the dual-arm robot system.
        """
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.visualizer = get_visualizer(
            list(self._cameras_ft.keys()), ['arm_left', 'arm_right'], config.draw_2d, config.draw_3d) \
            if config.visualize else None

        self._prepare_robots()

    def _prepare_robots(self) -> None:
        """
        Initialize left and right robot arm controllers.
        This method must be implemented by subclasses to create and configure:
        - self.left_robot: Controller for left arm (instance of BaseRobot or similar)
        - self.right_robot: Controller for right arm (instance of BaseRobot or similar)
        """
        raise NotImplementedError

    @property
    def _motors_ft(self) -> Dict[str, Any]:
        """
        Define motor feature types for both arms.
        Combines features from left and right arms with prefixes.
        Returns:
        - motors_dict: Combined motor feature types with 'left_' and 'right_' prefixes.
        """
        left_ft = {f"left_{each}": float for each in self.left_robot._motors_ft.keys()}
        right_ft = {f"right_{each}": float for each in self.right_robot._motors_ft.keys()}
        return {**left_ft, **right_ft}
    
    @property
    def _cameras_ft(self) -> Dict[str, tuple]:
        """
        Define camera feature shapes for observation space.
        Returns:
        - cameras_dict: Camera feature shapes keyed by camera names.
        """
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        """
        Combine motor and camera features for observation space.
        Returns:
        - obs_dict: Combined observation features from motors and cameras.
        """
        return {**self._motors_ft, **self._cameras_ft}
    
    @cached_property
    def action_features(self) -> Dict[str, Any]:
        """
        Define action space features for both arms.
        Returns:
        - action_dict: Combined action feature types with 'left_' and 'right_' prefixes.
        """
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:
        """
        Check if both arms and all cameras are connected.
        Returns:
        - connected: True if both arms and all cameras are connected, False otherwise.
        """
        return self.left_robot.is_connected and self.right_robot.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )
    
    def get_joint_state(self) -> np.ndarray:
        """
        Get current joint states for both arms.
        Returns:
        - state: Concatenated joint states from left and right arms.
        """
        state_left = self.left_robot.get_joint_state()
        state_right = self.right_robot.get_joint_state()
        return np.concatenate([state_left, state_right])
    
    def set_joint_state(self, state: np.ndarray):
        """
        Set joint positions for both arms.
        Params:
        - state: Concatenated joint states for left and right arms.
        """
        n_left = len(self.left_robot._motors_ft)
        self.left_robot.set_joint_state(state[:n_left])
        self.right_robot.set_joint_state(state[n_left:])
    
    def get_ee_state(self) -> np.ndarray:
        """
        Get current end-effector states for both arms.
        Returns:
        - state: Concatenated end-effector states from left and right arms.
        """
        state_left = self.left_robot.get_ee_state()
        state_right = self.right_robot.get_ee_state()
        return np.concatenate([state_left, state_right])
    
    def set_ee_state(self, state: np.ndarray):
        """
        Set end-effector poses for both arms.
        Params:
        - state: Concatenated end-effector states for left and right arms.
        """
        n_left = 7 
        self.left_robot.set_ee_state(state[:n_left])
        self.right_robot.set_ee_state(state[n_left:])
    
    def connect(self) -> None:
        """
        Connect to all robot components (both arms and cameras).
        """
        for cam in self.cameras.values():
            cam.connect()
        self.left_robot.connect()
        self.right_robot.connect()

        # Warmup cameras by capturing initial frames
        # This helps stabilize camera feeds and clear initial artifacts
        if self.cameras:
            for _ in range(10):
                for cam in self.cameras.values():
                    cam.async_read()
    
    def is_calibrated(self) -> bool:
        """
        Check if both arms are calibrated.
        Returns:
        - calibrated: True if both arms are calibrated, False otherwise.
        """
        return self.left_robot.is_calibrated() and self.right_robot.is_calibrated()
    
    def calibrate(self) -> None:
        """
        Calibrate both robot arms.
        """
        self.left_robot.calibrate()
        self.right_robot.calibrate()
    
    def configure(self) -> None:
        """
        Configure both robot arms.
        """
        self.left_robot.configure()
        self.right_robot.configure()
    
    def visualize(self) -> None:
        """
        Update visualization with current states and camera images.
        Shows both arms' end-effector states and camera feeds.
        """
        state_left = self.left_robot.get_ee_state()
        state_right = self.right_robot.get_ee_state()
        observation = self.get_observation()
        images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
        self.visualizer.add(images, [state_left, state_right])
        self.visualizer.plot()
    
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send actions to both arms and return updated states.
        Params:
        - action: Dictionary of actions for both arms, prefixed with 'left_' and 'right_'.
        Returns:
        - state: Updated states from both arms, prefixed with 'left_' and 'right_'.
        """
        action_left = {k.replace('left_', ''): v for k, v in action.items() if k.startswith('left_')}
        action_right = {k.replace('right_', ''): v for k, v in action.items() if k.startswith('right_')}
        
        state_left = self.left_robot.send_action(action_left)
        state_right = self.right_robot.send_action(action_right)

        if self.visualizer:
            self.visualize()

        state_left = {f"left_{k}": v for k, v in zip(self.left_robot._motors_ft.keys(), state_left)}
        state_right = {f"right_{k}": v for k, v in zip(self.right_robot._motors_ft.keys(), state_right)}
        return {**state_left, **state_right}
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get observations from both arms and all cameras.
        Returns:
        - obs_dict: Combined observations from both arms and cameras, with appropriate prefixes.
        """
        state_left = self.left_robot.get_observation()
        state_right = self.right_robot.get_observation()

        state_left = {f"left_{k}": v for k, v in state_left.items()}
        state_right = {f"right_{k}": v for k, v in state_right.items()}
        obs_dict = {**state_left, **state_right}

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            obs_dict[cam_key] = outputs

        return obs_dict
    
    def disconnect(self) -> None:
        """
        Disconnect from all robot components (both arms and cameras).
        """
        self.left_robot.disconnect()
        self.right_robot.disconnect()
        
        for cam in self.cameras.values():
            cam.disconnect()