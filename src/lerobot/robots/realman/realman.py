from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_realman import RealmanConfig


class Realman(Robot):
    """
    Realman is a robot class for controlling the Realman robot using joint control.

    Example:
        ```python
        config = RealmanConfig(
            dev_mode=65, ip="192.168.1.18", 
            cameras={"front": {"type": "dummy_camera", "height": 480, "width": 640, "fps": 30}}
        )
        robot = Realman(config)
        robot.connect()

        # get observation
        observation = robot.get_observation()

        # send action
        action = {"joint_1_pos": 0, "joint_2_pos": 10, "joint_3_pos": 20, 
                  "joint_4_pos": 30, "joint_5_pos": 40, "joint_6_pos": 50, 
                  "joint_7_pos": 60, "gripper_pos": 1000}
        robot.send_action(action)
        
        robot.disconnect()
        ```
    """

    config_class = RealmanConfig
    name = "realman"

    def __init__(self, config: RealmanConfig):
        try:
            from Robotic_Arm.rm_robot_interface import (
                RoboticArm, 
                rm_thread_mode_e,
            )
        except ImportError:
            raise ImportError("Realman robot requires the Robotic_Arm package. "
                              "Please install it using 'pip install Robotic_Arm'.")

        super().__init__(config)

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        self.config = config
        self.init_state = config.init_ee_state
        self.cameras = make_cameras_from_configs(config.cameras)
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f'{each}_pos': float for each in [
                'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper'
            ]
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
    
    def connect(self):
        self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        self.arm.rm_set_arm_run_mode(1)

        if self.config.init_state_type == 'joint':
            self._set_joint_state(self.config.init_state)
        elif self.config.init_state_type == 'end_effector':
            self._set_ee_state(self.config.init_state)
        else:
            raise ValueError(f"Unknown init_state_type: {self.config.init_state_type}")
        
        for cam in self.cameras.values():
            cam.connect()
    
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        print("Realman robot does not require calibration.")

    def configure(self):
        print("Realman robot does not require configuration.")
    
    def _set_joint_state(self, state: list[int]):
        self.arm.rm_movej(state[:-1], v=30, r=0, connect=0, block=self.config.block)
    
    def _get_joint_state(self) -> list[int]:
        # WARN: rm_get_current_arm_state not working in Realman API
        # error_code, state = self.arm.rm_get_current_arm_state()
        # joint = state['joint']
        # if error_code != 0:
        #     raise RuntimeError(f"Failed to get joint state: {error_code}")
        res = self.arm.rm_get_arm_current_trajectory()
        if res['return_code'] != 0:
            ret_code = res['return_code']
            raise RuntimeError(f'Failed to get joint state: {ret_code}')
        joint = res['data']
        ret_code, grip = self.arm.rm_get_gripper_state()
        grip = grip['actpos']
        if ret_code != 0:
            raise RuntimeError(f'Failed to get gripper state: {ret_code}')
        return joint + [grip]
    
    def _set_ee_state(self, state: list[int]):
        from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t
        # WARN: rm_movel not working in Realman API
        # self.arm.rm_movel(state[:6], v=30, r=0, connect=0, block=self.config.block)
        # self.arm.rm_set_gripper_position(int(state[6]), block=self.config.block)
        ret_code, joint = self.arm.rm_algo_inverse_kinematics(rm_inverse_kinematics_params_t(
            q_in=self._get_joint_state()[:-1],
            q_pose=state[:-1],
            flag=1
        ))
        if ret_code != 0:
            print('IK error:', ret_code)
        self._set_joint_state(joint + [state[-1]])

    def _get_ee_state(self) -> list[int]:
        # WARN: rm_get_current_arm_state not working in Realman API
        # error_code, _, pose, _ = self.arm.Get_Current_Arm_State()
        # error_code, state = self.arm.rm_get_current_arm_state()
        # pose = state['pose']
        # if error_code != 0:
        #     raise RuntimeError(f"Failed to get end-effector state: {error_code}")
        # error_code, grip = self.arm.Get_Gripper_State()
        # if error_code != 0:
        #     raise RuntimeError(f"Failed to get gripper state: {error_code}")
        # return pose + [grip]
        joint = self._get_joint_state()
        pose = self.arm.rm_algo_forward_kinematics(joint[:-1], flag=1)
        return pose + [joint[-1]]
    
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
            obs_dict[cam_key] = outputs

        print(obs_dict)
        return obs_dict
    
    def disconnect(self):
        ret_code = self.arm.rm_delete_robot_arm()
        if ret_code == 0:
            print("Realman robot disconnected.")
        else:
            raise RuntimeError("Failed to disconnect realman robot.")
