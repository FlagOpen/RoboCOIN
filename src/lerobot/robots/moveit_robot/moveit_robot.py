import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_moveit_robot import MoveitRobotConfig


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
        self.joint_names: list[str] = None

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        assert self.joint_names is not None, "Joint names are not set. Ensure the robot is connected."
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
        import rospy
        rospy.init_node('moveit_robot_node', anonymous=True)

        import moveit_commander
        moveit_commander.roscpp_initialize([])
        self.move_group = moveit_commander.MoveGroupCommander(self.config.move_group)

        self.joint_names = self.move_group.get_active_joints()

        if self.config.init_state_type == 'joint':
            self._set_joint_state(self.config.init_state)
        elif self.config.init_state_type == 'end_effector':
            self._set_ee_state(self.config.init_state)
        else:
            raise ValueError(f"Unknown init_state_type: {self.config.init_state_type}")
        time.sleep(1)
        
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
        success = self.move_group.go()
        if not success:
            print("Failed to set joint state")

    def _get_joint_state(self) -> list[int]:
        return self.move_group.get_current_joint_values()
    
    def _set_ee_state(self, state: list[int]):
        self.move_group.set_pose_target(state[:6])
        success, traj, _, _ = self.move_group.plan()
        if not success:
            print("Failed to plan end effector state")
            return
        joint = list(traj.joint_trajectory.points[-1].positions)
        if self.config.has_gripper and len(state) > 6:
            joint[-1] = state[-1]
            self.move_group.set_joint_value_target(joint)
        success = self.move_group.go()
        if not success:
            print("Failed to set end effector state")
            return

    def _get_ee_state(self) -> list[int]:
        xyz = self.move_group.get_current_pose().pose.position
        xyz = [xyz.x, xyz.y, xyz.z]
        rpy = self.move_group.get_current_rpy()
        state = xyz + rpy
        if self.config.has_gripper:
            gripper = self.move_group.get_current_joint_values()[-1]
            state += [gripper]
        else:
            state += [0.0]
        return state
    
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
        import moveit_commander
        moveit_commander.roscpp_shutdown()
        import rospy
        rospy.signal_shutdown('moveit_robot_node shutdown')
        print("Shutting down ROS node...")
        for cam in self.cameras.values():
            cam.disconnect()
