"""
Base class for robots controlled via end-effector (EE) commands.
"""

import numpy as np
from typing import Any, Dict

from lerobot.errors import DeviceNotConnectedError

from .base_robot import BaseRobot
from .configuration_base_robot import BaseRobotEndEffectorConfig
from .units_transform import UnitsTransform


class BaseRobotEndEffector(BaseRobot):
    """
    Base class for robots controlled via end-effector (EE) commands.
    Extends the BaseRobot class to provide end-effector level control.
    Handles unit conversions and action preparation specific to EE control.
    Params:
    - config: Configuration object for the end-effector robot
    e.g.
    ```python
    from lerobot.robots.base_robot.base_robot_end_effector import BaseRobotEndEffector
    from lerobot.robots.base_robot.configuration_base_robot import BaseRobotEndEffectorConfig

    config = BaseRobotEndEffectorConfig(
        pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
        model_pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
    )
    robot = BaseRobotEndEffector(config)
    robot.connect()
    obs = robot.get_observation()
    action = {'x': 0.1, 'y': 0.0, 'z': 0.2, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper': 0.05}
    robot.send_action(action)
    robot.disconnect()
    ```
    """

    config_class = BaseRobotEndEffectorConfig
    name = "base_robot_end_effector"

    def __init__(self, config: BaseRobotEndEffectorConfig) -> None:
        """
        Initialize the end-effector controlled robot.
        """
        super().__init__(config)
        self.model_pose_transform = UnitsTransform(config.model_pose_units)

    def prepare_and_send_action(self, action: np.ndarray) -> None:
        """
        Prepare end-effector action (delta vs absolute) and send to robot.
        Handles both absolute pose commands and relative delta commands.
        Params:
        - action: End-effector action values in standard units
        """
        if self.config.delta_with == 'previous':
            assert self._current_state is not None, "Current state is None, please run `get_observation` first."
            action += self._current_state
        elif self.config.delta_with == 'initial':
            assert self._init_state is not None, "Initial state is None, please run `connect` first."
            action += self._init_state
        self.set_ee_state(action)
    
    def connect(self) -> None:
        """
        Connect to the robot and initialize the initial end-effector state.
        """
        super().connect()
        self._init_state = self.get_ee_state()
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation and update EE state.
        Calls the base class method and updates current EE state.
        Returns:
        - obs_dict: Dictionary of current observations
        """
        obs_dict = super().get_observation()
        self._current_state = self.get_ee_state()
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> Dict[str, Any]:
        """
        Send end-effector action to robot and return updated state.
        Handles unit conversion and action preparation before sending to hardware.
        Params:
        - action: Dictionary of end-effector action values
        Returns:
        - state: Dictionary of updated joint states after action execution
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        action = np.array([action[each] for each in self.action_features.keys()])
        action = self.model_pose_transform.input_transform(action) # model -> standard

        action = self.prepare_and_send_action(action)

        if self.visualizer:
            self.visualize()

        state = self.get_joint_state()
        return {k: v for k, v in zip(self._motors_ft.keys(), state)}

    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Define the action features for end-effector control.
        Returns:
        - Dictionary mapping action feature names to their types
        """
        return {
            each: float for each in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        }