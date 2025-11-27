"""
Piper robot class implementation.
"""

import importlib
import numpy as np
import time

from ..base_robot import BaseRobot
from .configuration_piper import PiperConfig


class Piper(BaseRobot):
    """
    Piper robot class implementation.
    Params:
    - config: PiperConfig
    """

    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig) -> None:
        super().__init__(config)
        self.config = config
    
    def _check_dependency(self) -> None:
        """
        Check if the piper_sdk package is installed.
        Raises ImportError if not installed.
        """
        if importlib.util.find_spec("piper_sdk") is None:
            raise ImportError(
                "Piper robot requires the piper_sdk package. "
                "Please install it using 'pip install piper_sdk'."
            )
    
    def _connect_arm(self) -> None:
        """
        Connect to the Piper robotic arm.
        Initializes the C_PiperInterface_V2 interface and connects to the robot.
        """
        from piper_sdk import C_PiperInterface_V2
        self.arm = C_PiperInterface_V2(self.config.can)
        self.arm.ConnectPort()
        while not self.arm.EnablePiper():
            print("Waiting for Piper to enable...")
            time.sleep(0.1)
    
    def _disconnect_arm(self) -> None:
        """
        Disconnect from the Piper robotic arm.
        Ensures the arm is disconnected properly.
        """
        while self.arm.DisconnectPort():
            print("Waiting for Piper to disconnect...")
            time.sleep(0.1)
    
    def _set_joint_state(self, state: np.ndarray) -> None:
        """
        Set the joint state of the Piper robot.
        Use the Piper SDK to move the joints and set the gripper position.
        Params:
        - state: np.ndarray of joint positions
        """
        self.arm.MotionCtrl_2(0x01, 0x01, self.config.velocity, 0x00)
        self.arm.JointCtrl(*state[:6])
        self.arm.GripperCtrl(int(state[6]), 1000, 0x01, 0)
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get the joint state of the Piper robot.
        Use the Piper SDK to retrieve the current joint and gripper states.
        Returns:
        - state: np.ndarray of joint positions
        """
        joint_state = self.arm.GetArmJointMsgs().joint_state
        grip = self.arm.GetArmGripperMsgs().gripper_state.grippers_angle
        return [
            joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
            joint_state.joint_4, joint_state.joint_5, joint_state.joint_6,
            grip
        ]
    
    def _set_ee_state(self, state: np.ndarray) -> None:
        """
        Set the end-effector state of the Piper robot.
        Uses the Piper SDK to set the end-effector pose and gripper position.
        Params:
        - state: np.ndarray of end-effector positions
        """
        self.arm.MotionCtrl_2(0x01, 0x00, self.config.velocity, 0x00)
        self.arm.EndPoseCtrl(*state[:6])
        self.arm.GripperCtrl(int(state[6]), 1000, 0x01, 0)

    def _get_ee_state(self) -> np.ndarray:
        """
        Get the end-effector state of the Piper robot.
        Uses the Piper SDK to retrieve the current end-effector pose and gripper position.
        Returns:
        - state: np.ndarray of end-effector positions
        """
        end_pose = self.arm.GetArmEndPoseMsgs().end_pose
        grip = self.arm.GetArmGripperMsgs().gripper_state.grippers_angle
        return [
            end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis,
            end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis,
            grip
        ]