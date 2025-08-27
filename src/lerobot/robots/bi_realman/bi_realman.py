from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_bi_realman import BiRealmanConfig
from ..realman.configuration_realman import RealmanConfig
from ..realman.realman import Realman


class BiRealman(Robot):
    """
    BiRealman is a robot class for controlling the BiRealman robot using joint control.

    Example:
        ```python
        config = BiRealmanConfig(
            port_left="can1",
            port_right="can2",
            cameras={"front": {"type": "dummy_camera", "height": 480, "width": 640, "fps": 30}}
        )
        robot = BiRealman(config)
        robot.connect()

        # get observation
        observation = robot.get_observation()

        # send action
        action = {
            "left_joint_1.pos": 0, "left_joint_2.pos": 10, "left_joint_3.pos": 20,
            "left_joint_4.pos": 30, "left_joint_5.pos": 40, "left_joint_6.pos": 50, "left_gripper.pos": 60000,
            "right_joint_1.pos": 0, "right_joint_2.pos": 10, "right_joint_3.pos": 20,
            "right_joint_4.pos": 30, "right_joint_5.pos": 40, "right_joint_6.pos": 50, "right_gripper.pos": 60000
        }
        robot.send_action(action)

        robot.disconnect()
        ```
    """

    config_class = BiRealmanConfig
    name = "bi_realman"

    def __init__(self, config: BiRealmanConfig):
        super().__init__(config)

        if len(config.init_state) < 10:
            init_state_left = init_state_right = config.init_state
        else:
            init_state_left = config.init_state[:len(config.init_state) // 2]
            init_state_right = config.init_state[len(config.init_state) // 2:]
        
        left_arm_config = RealmanConfig(
            id=f"{config.id}_left" if config.id else None,
            ip=config.ip_left,
            port=config.port_left,
            cameras={},
            init_type=config.init_type,
            init_state=init_state_left,
            block=config.block,
        )
        right_arm_config = RealmanConfig(
            id=f"{config.id}_right" if config.id else None,
            ip=config.ip_right,
            port=config.port_right,
            cameras={},
            init_type=config.init_type,
            init_state=init_state_right,
            block=config.block,
        )

        self.left_arm = Realman(left_arm_config)
        self.right_arm = Realman(right_arm_config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
    
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
    
    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft
    
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
                    obs_dict[f"{cam_key}.{key}"] = value
            else:
                obs_dict[cam_key] = outputs

        return obs_dict
    
    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        print("BiRealman robot disconnected.")
