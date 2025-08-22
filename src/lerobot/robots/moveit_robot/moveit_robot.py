from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_moveit_robot import MoveitRobotConfig
from ..misc.transforms import (
    euler_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
)

class MoveitRobot(Robot):
    """
    RosRobot is a robot class for controlling the ROS robot using joint control.

    Example:
        ```python
        config = ROSRobotConfig(
            subscribers={
                "joint": {
                    "name": "/get_joint_states",
                    "data_class": JointState,
                    "queue_size": 10,
                },
            },
            publishers={
                "joint": {
                    "name": "/joint_states",
                    "data_class": JointState,
                    "queue_size": 10,
                },
            },
            cameras={
                "front": {
                    "type": "dummy_camera", 
                    "height": 480, 
                    "width": 640, 
                    "fps": 30
                }
            }
        )
        robot = ROSRobot(config)
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

    config_class = MoveitRobotConfig
    name = "moveit_robot"

    def __init__(self, config: MoveitRobotConfig):
        try:
            # import moveit_commander
            from moveit_commander import MoveGroupCommander
        except ImportError:
            raise ImportError("Moveit robot requires the moveit, ensure moveit is installed.")

        super().__init__(config)

        self.move_group: MoveGroupCommander = None

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.messages = {}
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f'{each}_pos': float for each in self.joint_names
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
        import moveit_commander
        moveit_commander.roscpp_initialize([])
        self.move_group = moveit_commander.MoveGroupCommander(self.config.move_group)

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
        print("ROS robot does not require calibration.")

    def configure(self):
        print("ROS robot does not require configuration.")
    
    def _set_joint_state(self, state: list[int]):
        self.move_group.set_joint_value_target(state)
        success = self.move_group.go(wait=True)
        if not success:
            print("Failed to set joint state")

    def _get_joint_state(self) -> list[int]:
        return self.move_group.get_current_joint_values()
    
    def _set_ee_state(self, state: list[int]):
        from moveit_commander import Pose
        pose = Pose()
        pose.position.x = state[0]
        pose.position.y = state[1]
        pose.position.z = state[2]
        quaternion = rotation_matrix_to_quaternion(
            euler_to_rotation_matrix(state[3:])
        )
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        self.move_group.set_pose_target(pose)

    def _get_ee_state(self) -> list[int]:
        pose = self.move_group.get_current_pose().pose
        return [
            pose.position.x, pose.position.y, pose.position.z,
            *rotation_matrix_to_euler(
                quaternion_to_rotation_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            )
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
            obs_dict[cam_key] = outputs

        return obs_dict
    
    def disconnect(self):
        for cam in self.cameras.values():
            cam.disconnect()
