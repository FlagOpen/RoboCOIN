from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_dummy import DummyConfig
from ..misc import get_standardization, get_transform, get_visualizer


class DummyRobot(Robot):
    """
    DummyRobot is a placeholder robot that simulates a robot's behavior without actual hardware.
    It is used for testing and development purposes, simulating a robot that always outputs the same

    Example:
        ```python
        config = DummyConfig(cameras={"front": {"type": "dummy_camera", "height": 480, "width": 640, "fps": 30}})
        robot = DummyRobot(config)
        robot.connect()
        
        # get observation
        observation = robot.get_observation()

        # send action
        action = {"x": 0.1, "y": 0.2, "z": 0.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "gripper": 0.5}
        robot.send_action(action)

        robot.disconnect()
        ```
    """

    config_class = DummyConfig
    name = "dummy"

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self._is_connected = False
        self._is_calibrated = False
        self.states = [config.init_ee_state]

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.standardization = get_standardization(self.name) if config.standardize else None
        self.transform = get_transform(config.control_mode, config.base_euler)
        self.visualizer = get_visualizer(list(self._cameras_ft.keys()), ['arm'], [config.init_ee_state], 'ee_absolute') \
                          if config.visualize else None
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            each: float for each in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        }
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        cameras_features = {}
        for cam_name, cam in self.cameras.items():
            if hasattr(cam, 'observation_features'):
                features = cam.observation_features
                cameras_features.update({f"{cam_name}.{k}": v for k, v in features.items()})
            else:
                cameras_features[cam_name] = (cam.height, cam.width, 3)
        return cameras_features
    
    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(self.camera.is_connected for self.camera in self.cameras.values())
    
    def connect(self):
        self._is_connected = True
        for camera in self.cameras.values():
            camera.connect()
        print("Dummy robot connected.")
    
    def is_calibrated(self) -> bool:
        return self._is_calibrated
    
    def calibrate(self):
        self._is_calibrated = True
        print("Dummy robot calibrated.")

    def configure(self):
        print("Dummy robot configured.")
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        assert all(k in action for k in self._motors_ft.keys()), \
            f"Action must contain keys: {list(self._motors_ft.keys())}, but got {list(action.keys())}"

        action = [action[each] for each in self._motors_ft.keys()]
        if self.standardization:
            action = self.standardization.output_transform(action)

        new_state = self.transform(self.states[-1], action)
        print(f'set state: {action}, new_state: {new_state}')
        self.states.append(new_state)

        if self.visualizer:
            observation = self.get_observation()
            images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
            self.visualizer.add(images, [new_state])
            self.visualizer.plot()
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        current_state = self.states[-1]
        if self.standardization:
            current_state = self.standardization.input_transform(current_state)
        
        obs_dict = {
            each: current_state[i] for i, each in enumerate(self._motors_ft.keys())
        }

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    obs_dict[f"{cam_key}.{key}"] = value
            else:
                obs_dict[cam_key] = outputs
        
        return obs_dict
    
    def disconnect(self):
        print("Dummy robot disconnected.")
        self._is_connected = False
        for camera in self.cameras.values():
            camera.disconnect()
