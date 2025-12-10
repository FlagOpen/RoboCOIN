"""
Piper leader class implementation.
"""

import importlib
import numpy as np
import time

from ..base_leader import BaseLeader
from .config_piper_leader import PiperLeaderConfig


class PiperLeader(BaseLeader):
    """
    Piper leader class implementation.
    Params:
    - config: PiperLeaderConfig
    """

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig) -> None:
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
        self.arm = C_PiperInterface_V2(self.config.can, judge_flag=True)
        # self.arm.MasterSlaveConfig(0xFA, 0, 0, 0)
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
    
    @property
    def is_connected(self) -> bool:
        """
        Check if the Piper robotic arm is connected.
        Returns:
        - is_connected: bool
        """
        return self.arm.get_connect_status()
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get the joint state of the Piper robot.
        Use the Piper SDK to retrieve the current joint and gripper states.
        Returns:
        - state: np.ndarray of joint positions
        """
        joint_state = self.arm.GetArmJointCtrl().joint_ctrl
        grip = self.arm.GetArmGripperCtrl().gripper_ctrl.grippers_angle
        return [
            joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
            joint_state.joint_4, joint_state.joint_5, joint_state.joint_6,
            grip
        ]

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