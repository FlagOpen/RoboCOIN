import numpy as np
from typing import Any

from lerobot.errors import DeviceNotConnectedError

from .base_robot import BaseRobot
from .configuration_base_robot import BaseRobotEndEffectorConfig
from .units_transform import UnitsTransform


class BaseRobotEndEffector(BaseRobot):

    config_class = BaseRobotEndEffectorConfig
    name = "base_robot_end_effector"

    def __init__(self, config: BaseRobotEndEffectorConfig):
        super().__init__(config)
        self.ee_transform = UnitsTransform(config.pose_units)
        self.model_pose_transform = UnitsTransform(config.model_pose_units)

    def prepare_and_send_action(self, action: np.ndarray) -> np.ndarray:
        if self.config.delta_with == 'previous':
            assert self._current_state is not None, "Current state is None, please run `get_observation` first."
            action += self._current_state
        elif self.config.delta_with == 'initial':
            assert self._init_state is not None, "Initial state is None, please run `connect` first."
            action += self._init_state
        self.set_ee_state(action)
    
    def connect(self):
        super().connect()
        self._init_state = self.get_ee_state()
    
    def get_observation(self) -> dict[str, Any]:
        obs_dict = super().get_observation()
        self._current_state = self.get_ee_state()
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        action = np.array([action[each] for each in self.action_features.keys()])
        # model -> standard
        action = self.model_pose_transform.input_transform(action)

        action = self.prepare_and_send_action(action)

        if self.visualizer:
            self.visualize()

        state = self.get_joint_state()
        return {k: v for k, v in zip(self._motors_ft.keys(), state)}

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            each: float for each in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        }