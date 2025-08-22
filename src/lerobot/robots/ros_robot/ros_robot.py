from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_ros_robot import ROSRobotConfig
from ..misc.transforms import (
    euler_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
)

class ROSRobot(Robot):
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

    config_class = ROSRobotConfig
    name = "ros_robot"

    def __init__(self, config: ROSRobotConfig):
        try:
            import rospy
        except ImportError:
            raise ImportError("ROS robot requires the rospy package. "
                              "Please install it using 'pip install rospy'.")

        super().__init__(config)

        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.messages = {}
    
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
        import rospy
        return (
            not rospy.is_shutdown() and
            all(self.camera.is_connected for self.camera in self.cameras.values())
        )
    
    def connect(self):
        import rospy
        rospy.init_node('ros_robot', anonymous=True)
        # each subscriber can be configured in the config file as a dict
        # e.g.
        # config.subscribers: [
        #     'joint': {
        #         'name': '/joint_states',
        #         'data_class': JointState,
        #         'queue_size': 10,
        #     }
        # ]
        self.subscribers = {
            sub_name: rospy.Subscriber(**sub_config, callback=lambda msg: self.messages.update({sub_name: msg}))
            for sub_name, sub_config in self.config.subscribers.items()
        }
        # each publisher can be configured in the config file as a dict
        # e.g.
        # config.publishers: [
        #     'joint': {
        #         'name': '/joint_states',
        #         'data_class': JointState,
        #         'queue_size': 10,
        #     }
        # ]
        self.publishers = {
            pub_name: rospy.Publisher(**pub_config)
            for pub_name, pub_config in self.config.publishers.items()
        }

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
        import rospy
        assert 'joint' in self.publishers, "Joint state publisher not configured."
        from sensor_msgs.msg import JointState
        msg = JointState()
        msg.name = self.config.joint_names
        msg.position = state
        msg.header.stamp = rospy.Time.now()
        self.publishers['joint'].publish(msg)
        rospy.sleep(0.1)  # wait for the message to be sent

    def _get_joint_state(self) -> list[int]:
        assert 'joint' in self.subscribers, "Joint state subscriber not configured."
        msg = self.messages['joint']
        return list(msg.position)
    
    def _set_ee_state(self, state: list[int]):
        import rospy
        assert 'pose' in self.publishers, "End-effector state publisher not configured."
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.pose.position.x = state[0]
        msg.pose.position.y = state[1]
        msg.pose.position.z = state[2]

        quat = rotation_matrix_to_quaternion(
            euler_to_rotation_matrix(state[3], state[4], state[5])
        )
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        msg.header.stamp = rospy.Time.now()
        self.publishers['pose'].publish(msg)

    def _get_ee_state(self) -> list[int]:
        assert 'pose' in self.subscribers, "End-effector state subscriber not configured."
        msg = self.messages['pose']
        position = msg.pose.position
        orientation = msg.pose.orientation
        euler = rotation_matrix_to_euler(
            quaternion_to_rotation_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
        )
        return [position.x, position.y, position.z, euler[0], euler[1], euler[2]]
    
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
        for sub in self.subscribers.values():
            sub.unregister()
        for pub in self.publishers.values():
            pub.unregister()
        import rospy
        rospy.signal_shutdown('ROS robot disconnected.')
        for cam in self.cameras.values():
            cam.disconnect()
