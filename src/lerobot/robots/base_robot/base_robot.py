import numpy as np
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_base_robot import BaseRobotConfig
from .units_transform import UnitsTransform
from .visualization import get_visualizer


class BaseRobot(Robot):

    config_class = BaseRobotConfig
    name = "base_robot"

    def __init__(self, config: BaseRobotConfig):
        super().__init__(config)
        self._check_dependency()

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.arm = None
        self.visualizer = get_visualizer(
            list(self._cameras_ft.keys()), ['arm'], config.draw_2d, config.draw_3d) \
            if config.visualize else None

        self.model_transform = UnitsTransform(config.model_units)
        self.joint_transform = UnitsTransform(config.joint_units)
        self.pose_transform = UnitsTransform(config.pose_units)


        self._init_state = None
        self._current_state = None
    
    def _check_dependency(self):
        return
    
    def _connect_arm(self):
        raise NotImplementedError
    
    def _disconnect_arm(self):
        raise NotImplementedError
    
    def _set_joint_state(self, state: np.ndarray):
        raise NotImplementedError
    
    def _get_joint_state(self) -> np.ndarray:
        raise NotImplementedError

    def set_joint_state(self, state: np.ndarray):
        # standard -> joint
        state = self.joint_transform.output_transform(state)
        self._set_joint_state(state)
    
    def get_joint_state(self) -> np.ndarray:
        state = self._get_joint_state()
        # joint -> standard
        return self.joint_transform.input_transform(state)
    
    def get_ee_state(self) -> np.ndarray:
        state = self._get_ee_state()
        # end_effector -> standard
        return self.pose_transform.input_transform(state)

    def set_ee_state(self, state: np.ndarray):
        # standard -> end_effector
        state = self.pose_transform.output_transform(state)
        self._set_ee_state(state)
    
    def prepare_and_send_action(self, action: np.ndarray) -> np.ndarray:
        if self.config.delta_with == 'previous':
            assert self._current_state is not None, "Current state is None, please run `get_observation` first."
            action += self._current_state
        elif self.config.delta_with == 'initial':
            assert self._init_state is not None, "Initial state is None, please run `connect` first."
            action += self._init_state
        self.set_joint_state(action)
    
    def visualize(self):
        state = self.get_ee_state()
        observation = self.get_observation()
        images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
        self.visualizer.add(images, [state])
        self.visualizer.plot()
    
    def connect(self):
        for cam in self.cameras.values():
            cam.connect()
        self._connect_arm()

        # warmup camera
        for _ in range(10):
            self.get_observation()

        if self.config.init_type == 'joint':
            self.set_joint_state(self.config.init_state)
        elif self.config.init_type == 'end_effector':
            self.set_ee_state(self.config.init_ee_state)
        self._init_state = self.get_joint_state()

    def disconnect(self):
        for cam in self.cameras.values():
            cam.disconnect()
        self._disconnect_arm()
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        action = np.array([action[each] for each in self._motors_ft.keys()])
        # model -> standard
        action = self.model_transform.input_transform(action)

        action = self.prepare_and_send_action(action)

        if self.visualizer:
            self.visualize()

        state = self.get_joint_state()
        return {k: v for k, v in zip(self._motors_ft.keys(), state)}
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state = self.get_joint_state()
        # standard -> model
        state_to_send = self.model_transform.output_transform(state)
        obs_dict = {k: v for k, v in zip(self._motors_ft.keys(), state_to_send)}

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            obs_dict[cam_key] = outputs

        self._current_state = state

        return obs_dict
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f'{each}_pos': float for each in self.config.joint_names
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            all(self.camera.is_connected for self.camera in self.cameras.values())
        )
    
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        pass

    def configure(self):
        pass