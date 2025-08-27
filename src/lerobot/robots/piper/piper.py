import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_piper import PiperConfig


class Piper(Robot):
    """
    Piper is a robot class for controlling the Piper robot using joint control.

    Example:
        ```python
        config = PiperConfig(port="can1", cameras={"front": {"type": "dummy_camera", "height": 480, "width": 640, "fps": 30}})
        robot = Piper(config)
        robot.connect()

        # get observation
        observation = robot.get_observation()

        # send action
        action = {"joint_1_pos": 0, "joint_2_pos": 10, "joint_3_pos": 20, 
                  "joint_4_pos": 30, "joint_5_pos": 40, "joint_6_pos": 50, 
                  "gripper_pos": 60000}
        robot.send_action(action)
        
        robot.disconnect()
        ```
    """

    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError:
            raise ImportError("Piper robot requires the piper_sdk package. "
                              "Please install it using 'pip install piper_sdk'.")

        super().__init__(config)

        self.config = config
        self.arm = C_PiperInterface_V2(config.port)
        self.cameras = make_cameras_from_configs(config.cameras)
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f'{each}_pos': float for each in ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
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
        return all(self.camera.is_connected for self.camera in self.cameras.values())
    
    def connect(self):
        self.arm.ConnectPort()
        while not self.arm.EnablePiper():
            print("Waiting for Piper to enable...")
            time.sleep(0.1)
        
        if self.config.init_state_type == 'joint':
            self._set_joint_state(self.config.init_state)
        elif self.config.init_state_type == 'end_effector':
            self._set_ee_state(self.config.init_state)
        else:
            raise ValueError(f"Unknown init_state_type: {self.config.init_state_type}")

        print("Piper robot connected.")
        
        for cam in self.cameras.values():
            cam.connect()
    
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        print("Piper robot does not require calibration.")

    def configure(self):
        print("Piper robot does not require configuration.")
    
    def _set_joint_state(self, state: list[int]):
        self.arm.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.arm.JointCtrl(*state[:6])
        self.arm.GripperCtrl(int(state[6]), 1000, 0x01, 0)
    
    def _get_joint_state(self) -> list[int]:
        joint_state = self.arm.GetArmJointMsgs().joint_state
        grip = self.arm.GetArmGripperMsgs().gripper_state.grippers_angle
        return [
            joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
            joint_state.joint_4, joint_state.joint_5, joint_state.joint_6,
            grip
        ]
    
    def _set_ee_state(self, state: list[int]):
        self.arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.arm.EndPoseCtrl(*state[:6])
        self.arm.GripperCtrl(int(state[6]), 1000, 0x01, 0)

    def _get_ee_state(self) -> list[int]:
        end_pose = self.arm.GetArmEndPoseMsgs().end_pose
        grip = self.arm.GetArmGripperMsgs().gripper_state.grippers_angle
        return [
            end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis,
            end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis,
            grip
        ]
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        self._set_joint_state([action[each] for each in self._motors_ft.keys()])
        state = self._get_joint_state()
        return {k: v for k, v in zip(self._motors_ft.keys(), state)}
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state = self._get_joint_state()
        obs_dict = {k: v for k, v in zip(self._motors_ft.keys(), state)}

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    obs_dict[f"{cam_key}_{key}"] = value
            else:
                obs_dict[cam_key] = outputs

        return obs_dict
    
    def disconnect(self):
        while self.arm.DisconnectPort():
            print("Waiting for Piper to disconnect...")
            time.sleep(0.1)
        print("Piper robot disconnected.")
