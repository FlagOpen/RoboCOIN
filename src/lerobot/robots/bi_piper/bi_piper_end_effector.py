from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_bi_piper import BiPiperEndEffectorConfig
from ..piper import PiperEndEffectorConfig
from ..piper import PiperEndEffector
from ..misc import get_visualizer


class BiPiperEndEffector(Robot):
    """
    BiPiperEndEffector is a robot class for controlling the end effector of the BiPiper robot using end-effector control.

    Example:
        ```python
        config = BiPiperEndEffectorConfig(
            port_left="can1",
            port_right="can2",
            cameras={"front": {"type": "dummy_camera", "height": 480, "width": 640, "fps": 30}}
        )
        robot = BiPiperEndEffector(config)
        robot.connect()

        # get observation
        observation = robot.get_observation()

        # send action
        action = {
            "left_x": 0.1, "left_y": 0.2, "left_z": 0.3,
            "left_roll": 0.0, "left_pitch": 0.0, "left_yaw": 0.0, "left_gripper": 0.5,
            "right_x": 0.1, "right_y": 0.2, "right_z": 0.3,
            "right_roll": 0.0, "right_pitch": 0.0, "right_yaw": 0.0, "right_gripper": 0.5
        }
        robot.send_action(action)

        robot.disconnect()
        ```
    """
    
    config_class =  BiPiperEndEffectorConfig
    name = "bi_piper_end_effector"

    def __init__(self, config: BiPiperEndEffectorConfig):
        super().__init__(config)

        left_arm_config = PiperEndEffectorConfig(
            id=f"{config.id}_left" if config.id else None,
            port=config.port_left,
            cameras={},
            init_ee_state=config.init_ee_state,
            control_mode=config.control_mode,
            delta_with_previous=config.delta_with_previous,
            base_euler=config.base_euler,
            visualize=False,
        )
        right_arm_config = PiperEndEffectorConfig(
            id=f"{config.id}_right" if config.id else None,
            port=config.port_right,
            cameras={},
            init_ee_state=config.init_ee_state,
            control_mode=config.control_mode,
            delta_with_previous=config.delta_with_previous,
            base_euler=config.base_euler,
            visualize=False,
        )

        self.left_arm = PiperEndEffector(left_arm_config)
        self.right_arm = PiperEndEffector(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.visualizer = get_visualizer(list(self._cameras_ft.keys()), 
                                         ['arm_left', 'arm_right'], 
                                         [self.left_arm.standardization.input_transform(config.init_ee_state), 
                                          self.right_arm.standardization.input_transform(config.init_ee_state)], 
                                         'ee_absolute') \
                          if config.visualize else None
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        left_ft = {f"left_{each}": float for each in self.left_arm._motors_ft.keys()}
        right_ft = {f"right_{each}": float for each in self.right_arm._motors_ft.keys()}
        return {**left_ft, **right_ft}
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict[str, Any]:
        return {
            each: float for each in [
                'left_x', 'left_y', 'left_z', 'left_roll', 'left_pitch', 'left_yaw', 'left_gripper',
                'right_x', 'right_y', 'right_z', 'right_roll', 'right_pitch', 'right_yaw', 'right_gripper',
            ]
        }
    
    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected and all(cam.is_connected for cam in self.cameras.values())
    
    def connect(self):
        self.left_arm.connect()
        self.right_arm.connect()

        for cam in self.cameras.values():
            cam.connect()
    
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated
    
    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()
    
    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        if self.visualizer:
            left_state = self.left_arm.standardization.input_transform(self.left_arm._get_ee_state())
            right_state = self.right_arm.standardization.input_transform(self.right_arm._get_ee_state())
            observation = self.get_observation()
            images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
            self.visualizer.add(images, [left_state, right_state])
            self.visualizer.plot()

        send_action_left = {f"left_{k}": v for k, v in send_action_left.items()}
        send_action_right = {f"right_{k}": v for k, v in send_action_right.items()}
        return {**send_action_left, **send_action_right}
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    obs_dict[f"{cam_key}_{key}"] = value
            else:
                obs_dict[cam_key] = outputs
        return obs_dict
    
    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        print("BiPiper robot disconnected.")
