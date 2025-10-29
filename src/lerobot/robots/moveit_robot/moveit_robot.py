"""
Moveit robot class for controlling the robot using Moveit.
"""

import importlib
import numpy as np

from ..base_robot import BaseRobot
from .configuration_moveit_robot import MoveitRobotConfig


class MoveitRobot(BaseRobot):
    """
    Piper robot class implementation.
    Params:
    - config: PiperConfig
    """

    config_class = MoveitRobotConfig
    name = "moveit_robot"

    def __init__(self, config: MoveitRobotConfig) -> None:
        super().__init__(config)
        self.config = config
    
    def _check_dependency(self) -> None:
        """
        Check if the moveit_commander and rospy packages are installed.
        Raises ImportError if not installed.
        """
        if importlib.util.find_spec("moveit_commander") is None:
            raise ImportError(
                "Moveit robot requires the moveit_commander package. "
                "Please install it using 'pip install moveit_commander'."
            )
        if importlib.util.find_spec("rospy") is None:
            raise ImportError(
                "Moveit robot requires the rospy package. "
                "Please install it using 'pip install rospy'."
            )
    
    def _connect_arm(self) -> None:
        """
        Connect to the Moveit robotic arm.
        Initializes the MoveGroupCommander interface and connects to the robot.
        """
        import moveit_commander
        import rospy
        rospy.init_node('moveit_robot_node', anonymous=True)
        moveit_commander.roscpp_initialize([])
        self.move_group = moveit_commander.MoveGroupCommander(self.config.move_group)
        self.joint_names = self.move_group.get_active_joints()
    
    def _disconnect_arm(self) -> None:
        """
        Disconnect from the Moveit robotic arm.
        Ensures the arm is disconnected properly.
        """
        import moveit_commander
        moveit_commander.roscpp_shutdown()
        import rospy
        rospy.signal_shutdown('moveit_robot_node shutdown')
    
    def _set_joint_state(self, state: np.ndarray) -> None:
        """
        Set the joint state of the Moveit robot.
        Params:
        - state: np.ndarray, joint state to set
        """
        state = list(state)
        self.move_group.set_joint_value_target(state)
        success = self.move_group.go()
        if not success:
            raise RuntimeError("Failed to set joint state")

    def _get_joint_state(self) -> np.ndarray:
        """
        Get the current joint state of the Moveit robot.
        Returns:
        - state: np.ndarray, current joint state
        """
        return np.array(self.move_group.get_current_joint_values())
    
    def _set_ee_state(self, state: np.ndarray) -> None:
        """
        Set the end effector state of the Moveit robot.
        Params:
        - state: np.ndarray, end effector state to set
        """
        state = list(state)
        self.move_group.set_pose_target(state[:6])
        success, traj, _, _ = self.move_group.plan()
        if not success:
            raise RuntimeError("Failed to plan end effector state")

        # if state length > 6, the last element is the gripper state
        if self.config.has_gripper and len(state) > 6:
            joint = list(traj.joint_trajectory.points[-1].positions)
            joint[-1] = state[-1]
            self.move_group.set_joint_value_target(joint)
        success = self.move_group.go()
        if not success:
            raise RuntimeError("Failed to set end effector state")

    def _get_ee_state(self) -> np.ndarray:
        """
        Get the current end effector state of the Moveit robot.
        Returns:
        - state: np.ndarray, current end effector state
        """
        xyz = self.move_group.get_current_pose().pose.position
        xyz = [xyz.x, xyz.y, xyz.z]
        rpy = self.move_group.get_current_rpy()
        state = xyz + rpy
        if self.config.has_gripper:
            gripper = self.move_group.get_current_joint_values()[-1]
            state += [gripper]
        return np.array(state)