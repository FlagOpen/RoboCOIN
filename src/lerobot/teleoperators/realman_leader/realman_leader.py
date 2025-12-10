"""
Realman Leader teleoperator class implementation.
"""

import importlib
import numpy as np
from ..base_leader import BaseLeader
from .config_realman_leader import RealmanLeaderConfig


class RealmanLeader(BaseLeader):
    """
    Realman Leader teleoperator class implementation.
    Params:
    - config: RealmanLeaderConfig
    """

    config_class = RealmanLeaderConfig
    name = "realman_leader"

    def __init__(self, config: RealmanLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
    
    def _check_dependency(self) -> None:
        """
        Check for dependencies required by the Realman robot.
        Raises ImportError if the required package is not found.
        """
        if importlib.util.find_spec("Robotic_Arm") is None:
            raise ImportError(
                "Realman robot requires the Robotic_Arm package. "
                "Please install it using 'pip install Robotic_Arm'."
            )
    
    def _connect_arm(self) -> None:
        """
        Connect to the Realman robot arm.
        Initializes the RoboticArm interface and creates a robot arm handle.
        """
        from Robotic_Arm.rm_robot_interface import (
            RoboticArm, 
            rm_thread_mode_e,
        )
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        self.arm.rm_set_arm_run_mode(1)
    
    def _disconnect_arm(self) -> None:
        """
        Disconnect from the Realman robot arm.
        Destroys the robot arm handle.
        """
        ret_code = self.arm.rm_destroy()
        if ret_code != 0:
            raise RuntimeError(f'Failed to disconnect: {ret_code}')
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get the joint state of the Realman robot.
        Uses the RoboticArm interface to retrieve the current joint and gripper states.
        Raises RuntimeError if retrieval fails.
        Returns:
        - state: np.ndarray of joint positions
        """
        ret_code, joint = self.arm.rm_get_joint_degree()
        if ret_code != 0:
            raise RuntimeError(f'Failed to get joint state: {ret_code}')
        ret_code, grip = self.arm.rm_get_gripper_state()
        grip = grip['actpos']
        if ret_code != 0:
            raise RuntimeError(f'Failed to get gripper state: {ret_code}')
        return np.array(joint + [grip])

    def _get_ee_state(self) -> np.ndarray:
        """
        Get the end-effector state of the Realman robot.
        Uses the RoboticArm interface to compute forward kinematics based on current joint states.
        Raises RuntimeError if retrieval fails.
        Returns:
        - state: np.ndarray of end-effector positions
        """
        joint = self._get_joint_state()
        pose = self.arm.rm_algo_forward_kinematics(joint[:-1], flag=1)
        return np.array(pose + [joint[-1]])