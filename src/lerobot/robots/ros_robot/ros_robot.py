"""
ROS robot implementation.
"""

import importlib

from ..base_robot import BaseRobot
from .configuration_ros_robot import ROSRobotConfig


class RosRobot(BaseRobot):
    """
    ROS robot implementation.
    Params:
    - config: ROSRobotConfig
    """

    config_class = ROSRobotConfig
    name = "ros_robot"

    def __init__(self, config: ROSRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self.messages = {}

    def _check_dependency(self) -> None:
        """
        Check for dependencies required by the ROS robot.
        Raises ImportError if the required package is not found.
        """
        if importlib.util.find_spec("rospy") is None:
            raise ImportError(
                "ROS robot requires the rospy package. "
                "Please install it using 'pip install rospy'."
            )
    
    def _connect_arm(self) -> None:
        import rospy
        rospy.init_node('ros_robot', anonymous=True)
        self.subscribers = [
            rospy.Subscriber(
                **sub_config, 
                callback=lambda msg: self.messages.update({idx: msg})
            ) for idx, sub_config in enumerate(self.config.joint_subscribers)
        ]
        self.publishers = [
            rospy.Publisher(**pub_config)
            for pub_config in self.config.joint_publishers
        ]
    
    def _disconnect_arm(self) -> None:
        for sub in self.subscribers:
            sub.unregister()
        for pub in self.publishers:
            pub.unregister()
        import rospy
        rospy.signal_shutdown('ROS robot disconnected.')
    
    def _set_joint_state(self, state: list[int]):
        import rospy
        from sensor_msgs.msg import JointState
        if len(self.publishers) == 0:
            raise RuntimeError("No joint publishers configured for ROS robot.")
        elif len(self.publishers) == 1:
            msg = JointState()
            msg.name = self.config.joint_names
            msg.position = state
            msg.header.stamp = rospy.Time.now()
            self.publishers[0].publish(msg)
        else:
            assert len(state) == len(self.publishers), \
                "State length must match number of publishers."
            for i, pub in enumerate(self.publishers):
                msg = JointState()
                msg.name = [self.config.joint_names[i]]
                msg.position = [state[i]]
                msg.header.stamp = rospy.Time.now()
                pub.publish(msg)
        rospy.sleep(0.1)  # wait for the message to be sent

    def _get_joint_state(self) -> list[int]:
        if len(self.subscribers) == 0:
            raise RuntimeError("No joint subscribers configured for ROS robot.")
        elif len(self.subscribers) == 1:
            msg = self.messages.get(0, None)
            if msg is None:
                raise RuntimeError("No joint state message received yet.")
            return list(msg.position)
        else:
            state = []
            for i in range(len(self.subscribers)):
                msg = self.messages.get(i, None)
                if msg is None:
                    raise RuntimeError(f"No joint state message received yet for subscriber {i}.")
                state.append(msg.position[0])
            return state
    
    def _set_ee_state(self, state: list[int]):
        raise NotImplementedError("Setting end-effector state is not implemented for ROS robot.")

    def _get_ee_state(self) -> list[int]:
        raise NotImplementedError("Getting end-effector state is not implemented for ROS robot.")