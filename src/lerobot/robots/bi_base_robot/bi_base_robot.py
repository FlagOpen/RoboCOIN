import copy
import numpy as np
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .configuration_bi_base_robot import BiBaseRobotConfig
from ..base_robot import BaseRobot
from ..base_robot.visualization import get_visualizer


class BiBaseRobot(Robot):

    config_class = BiBaseRobotConfig
    name = "base_robot"

    def __init__(self, config: BiBaseRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.visualizer = get_visualizer(
            list(self._cameras_ft.keys()), ['arm_left', 'arm_right'], config.draw_2d, config.draw_3d) \
            if config.visualize else None

        self._prepare_robots()

    def _prepare_robots(self):
        raise NotImplementedError

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_ft = {f"left_{each}": float for each in self.left_robot._motors_ft.keys()}
        right_ft = {f"right_{each}": float for each in self.right_robot._motors_ft.keys()}
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
        return self.left_robot.is_connected and self.right_robot.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )
    
    def get_joint_state(self) -> np.ndarray:
        state_left = self.left_robot.get_joint_state()
        state_right = self.right_robot.get_joint_state()
        return np.concatenate([state_left, state_right])
    
    def set_joint_state(self, state: np.ndarray):
        n_left = len(self.left_robot._motors_ft)
        self.left_robot.set_joint_state(state[:n_left])
        self.right_robot.set_joint_state(state[n_left:])
    
    def get_ee_state(self) -> np.ndarray:
        state_left = self.left_robot.get_ee_state()
        state_right = self.right_robot.get_ee_state()
        return np.concatenate([state_left, state_right])
    
    def set_ee_state(self, state: np.ndarray):
        n_left = 7 
        self.left_robot.set_ee_state(state[:n_left])
        self.right_robot.set_ee_state(state[n_left:])
    
    def connect(self):
        self.left_robot.connect()
        self.right_robot.connect()
        
        for cam in self.cameras.values():
            cam.connect()
    
    def is_calibrated(self) -> bool:
        return self.left_robot.is_calibrated() and self.right_robot.is_calibrated()
    
    def calibrate(self):
        self.left_robot.calibrate()
        self.right_robot.calibrate()
    
    def configure(self):
        self.left_robot.configure()
        self.right_robot.configure()
    
    def visualize(self):
        state_left = self.left_robot.get_ee_state()
        state_right = self.right_robot.get_ee_state()
        observation = self.get_observation()
        images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
        self.visualizer.add(images, [state_left, state_right])
        self.visualizer.plot()
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_left = {k.replace('left_', ''): v for k, v in action.items() if k.startswith('left_')}
        action_right = {k.replace('right_', ''): v for k, v in action.items() if k.startswith('right_')}
        
        state_left = self.left_robot.send_action(action_left)
        state_right = self.right_robot.send_action(action_right)

        if self.visualizer:
            self.visualize()

        state_left = {f"left_{k}": v for k, v in zip(self.left_robot._motors_ft.keys(), state_left)}
        state_right = {f"right_{k}": v for k, v in zip(self.right_robot._motors_ft.keys(), state_right)}
        return {**state_left, **state_right}
    
    def get_observation(self) -> dict[str, Any]:
        state_left = self.left_robot.get_observation()
        state_right = self.right_robot.get_observation()

        state_left = {f"left_{k}": v for k, v in state_left.items()}
        state_right = {f"right_{k}": v for k, v in state_right.items()}
        obs_dict = {**state_left, **state_right}

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            obs_dict[cam_key] = outputs

        return obs_dict
    
    def disconnect(self):
        self.left_robot.disconnect()
        self.right_robot.disconnect()
        
        for cam in self.cameras.values():
            cam.disconnect()