import time
from ..base_robot import BaseRobot
from .configuration_realman import RealmanConfig


class Realman(BaseRobot):

    config_class = RealmanConfig
    name = "realman"

    def __init__(self, config: RealmanConfig):
        super().__init__(config)
        self.config = config

    def _check_dependency(self):
        try:
            from Robotic_Arm.rm_robot_interface import (
                RoboticArm, 
            )
        except ImportError:
            raise ImportError(
                "Realman robot requires the Robotic_Arm package. "
                "Please install it using 'pip install Robotic_Arm'."
            )
    
    def _connect_arm(self):
        from Robotic_Arm.rm_robot_interface import (
            RoboticArm, 
            rm_thread_mode_e,
        )
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        self.arm.rm_set_arm_run_mode(1)
    
    def _disconnect_arm(self):
        ret_code = self.arm.rm_destroy()
        if ret_code != 0:
            raise RuntimeError(f'Failed to disconnect: {ret_code}')
    
    def _set_joint_state(self, state: list[int]):
        success = self.arm.rm_movej(state[:-1], v=self.config.velocity, r=0, connect=0, block=self.config.block)
        if success != 0:
            raise RuntimeError(f'Failed movej')
        success = self.arm.rm_set_gripper_position(int(state[-1]), block=self.config.block, timeout=3)
        if success != 0:
            raise RuntimeError('Failed set gripper')
        if not self.config.block:
            time.sleep(self.config.wait_second)
    
    def _get_joint_state(self) -> list[int]:
        ret_code, joint = self.arm.rm_get_joint_degree()
        if ret_code != 0:
            raise RuntimeError(f'Failed to get joint state: {ret_code}')
        ret_code, grip = self.arm.rm_get_gripper_state()
        grip = grip['actpos']
        if ret_code != 0:
            raise RuntimeError(f'Failed to get gripper state: {ret_code}')
        return joint + [grip]
    
    def _set_ee_state(self, state: list[int]):
        from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t
        ret_code, joint = self.arm.rm_algo_inverse_kinematics(rm_inverse_kinematics_params_t(
            q_in=self._get_joint_state()[:-1],
            q_pose=state[:-1],
            flag=1
        ))
        if ret_code != 0:
            print('IK error:', ret_code)
        self._set_joint_state(joint + [state[-1]])

    def _get_ee_state(self) -> list[int]:
        joint = self._get_joint_state()
        pose = self.arm.rm_algo_forward_kinematics(joint[:-1], flag=1)
        return pose + [joint[-1]]