"""
Realman robot implementation.
"""

import importlib
import numpy as np
import time
from ..base_robot import BaseRobot
from .configuration_realman import RealmanConfig


class Realman(BaseRobot):
    """
    Realman robot implementation.
    Params:
    - config: RealmanConfig
    """

    config_class = RealmanConfig
    name = "realman"

    def __init__(self, config: RealmanConfig) -> None:
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
    
    def _set_joint_state(self, state: np.ndarray) -> None:
        """
        Set the joint state of the Realman robot.
        Uses the RoboticArm interface to move the joints and set the gripper position.
        Raises RuntimeError if the movement fails.
        Params:
        - state: np.ndarray of joint positions
        """
        state = list(state)
        success = self.arm.rm_movej(state[:-1], v=self.config.velocity, r=0, connect=0, block=self.config.block)

        if success != 0:
            raise RuntimeError(f'Failed movej')
        success = self.arm.rm_set_gripper_position(int(state[-1]), block=self.config.block, timeout=3)
        if success != 0:
            raise RuntimeError('Failed set gripper')

        if not self.config.block:
            time.sleep(self.config.wait_second)
    
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
    
    def _set_ee_state(self, state: np.ndarray) -> None:
        """
        Set the end-effector state of the Realman robot.
        Uses the RoboticArm interface to compute inverse kinematics and set joint states accordingly.
        Raises RuntimeError if inverse kinematics fails.
        Params:
        - state: np.ndarray of end-effector positions
        """
        from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t
        state = list(state)
        ret_code, joint = self.arm.rm_algo_inverse_kinematics(rm_inverse_kinematics_params_t(
            q_in=self._get_joint_state()[:-1],
            q_pose=state[:-1],
            flag=1
        ))
        if ret_code != 0:
            print('IK error:', ret_code)
        self._set_joint_state(joint + [state[-1]])

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