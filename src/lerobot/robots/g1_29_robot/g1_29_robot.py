from multiprocessing import Array, Lock
import numpy as np
import logging
from robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from robot_control.robot_hand_inspire_ftp import Inspire_Controller
from robot_control.robot_arm_ik_fk import G1_29_ArmIK_fk
from ..base_robot import BaseRobot
# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
import pinocchio as pin                             
from robot_control.robot_hand_dds import DexControl 
      
from pinocchio import casadi as cpin    
from pinocchio.visualize import MeshcatVisualizer   
logger = logging.getLogger(__name__)


class G1_29_Robot(BaseRobot):
    def __init__(self, config=None):
        self.config = config
        self.arm_controller = None
        self.hand_controller = None
        self.is_connected = False
        self.arm_ik_fk = G1_29_ArmIK_fk()

        self.left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        self.right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        self.dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        self.dc = DexControl(interfacenet="enp129s0")
    def _check_dependencys(self):
        """
        检查运行 G1 + Inspire 所需的依赖项
        """
        logger.info("Checking dependencies for G1_29_Robot...")

        #  检查 Unitree SDK 是否可导入
        try:
            import unitree_sdk2py
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except ImportError as e:
            raise RuntimeError("Unitree SDK 2.0 not installed or not in PYTHONPATH.") from e


    def _connect_arm(self, motion_mode=False, simulation_mode=False):
        """
        连接 G1 双臂控制器。
        """
        motion_mode = True
        logger.info("Connecting to G1 arm controller...")
        try:
            self.arm_controller = G1_29_ArmController(
                motion_mode=motion_mode,
                simulation_mode=simulation_mode
            )
            logger.info("G1 arm controller connected successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect arm controller: {e}") from e

    def _connect_hand(self, left_hand_array_in, right_hand_array_in,
                      dual_hand_data_lock=None,
                      dual_hand_state_array_out=None,
                      dual_hand_action_array_out=None,
                    simulation_mode=False):
        """
        连接 Inspire 手部控制器。
        """
        logger.info("Connecting to Inspire hand controller...")
        try:
            self.hand_controller = Inspire_Controller(
                left_hand_array_in=left_hand_array_in,
                right_hand_array_in=right_hand_array_in,
                dual_hand_data_lock=dual_hand_data_lock,
                dual_hand_state_array_out=dual_hand_state_array_out,
                dual_hand_action_array_out=dual_hand_action_array_out,
                simulation_mode=simulation_mode
            )
            logger.info("Inspire hand controller connected successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect hand controller: {e}") from e

    def _connect(self, 
                motion_mode: bool = False,
                simulation_mode: bool = False,
                left_hand_array_in=None,
                right_hand_array_in=None,
                dual_hand_data_lock=None,
                dual_hand_state_array_out=None,
                dual_hand_action_array_out=None,
              ) -> bool:
        """
        完整连接流程：
        1. 检查依赖
        2. 连接手
        3. 连接臂
        """
        try:
            self._check_dependencys()

            # 连接手部
            if left_hand_array_in is None or right_hand_array_in is None:
                logger.warning("Hand arrays not provided. Skipping hand connection.")
            else:
                self._connect_hand(
                    left_hand_array_in=left_hand_array_in,
                    right_hand_array_in=right_hand_array_in,
                    dual_hand_data_lock=dual_hand_data_lock,
                    dual_hand_state_array_out=dual_hand_state_array_out,
                    dual_hand_action_array_out=dual_hand_action_array_out,
                    simulation_mode=simulation_mode
                )

            # 连接手臂
            self._connect_arm(motion_mode=motion_mode, simulation_mode=simulation_mode)

            self.is_connected = True
            logger.info("G1_29_Robot fully connected.")
            return True

        except Exception as e:
            logger.error(f"Failed to connect G1_29_Robot: {e}")
            self.disconnect()
            return False

    def _disconnect_arm(self):
        if self.arm_controller is not None:
            self.arm_controller.ctrl_dual_arm_go_home()
            logger.info("Arm controller disconnected.")

    def _disconnect_hand(self):
        if self.hand_controller is not None:
            pass

    def _disconnect(self):
        """断开所有连接"""
        if not self.is_connected:
            return

        self._disconnect_hand()
        self._disconnect_arm()

        self.arm_controller = None
        self.hand_controller = None
        self.is_connected = False
        logger.info("G1_29_Robot disconnected.")

    def _get_joint_state(self):
        if not self.is_connected or self.arm_controller is None:
            return np.zeros(14)
        q = self.arm_controller.get_current_dual_arm_q()  
        return q
    
    def _set_joint_state(self, joint_state: np.ndarray):
        """
        设置机器人全部关节状态（臂 + 手），单位：弧度（按 joint_units 定义）。
        
        Args:
            joint_state (np.ndarray): shape=(28,), order:
                [L_arm(7), R_arm(7), L_hand(7), R_hand(7)]
                单位全部为 rad（Unitree SDK & DexControl 要求）
        
        Note:
            - 臂部：调用 self.arm_controller.ctrl_dual_arm(q, tau=0)
            - 手部：调用 self.dc.control_process(left_q, right_q)
        """
        if not self.is_connected or self.arm_controller is None:
            logger.warning(" Robot not connected. Skipping _set_joint_state.")
            return

        # 校验总维度
        if not isinstance(joint_state, np.ndarray) or joint_state.shape != (28,):
            logger.error(f"_set_joint_state: Expected (28,) array, got {joint_state.shape}")
            return

        # 按 joint_names 顺序（你已定义：臂14 + 左手7 + 右手7）
        ARM_START, ARM_END   = 0, 14
        L_HAND_START, L_HAND_END = 14, 21
        R_HAND_START, R_HAND_END = 21, 28

        q_arm   = joint_state[ARM_START:ARM_END]      # (14,)
        q_lhand = joint_state[L_HAND_START:L_HAND_END] # (7,)
        q_rhand = joint_state[R_HAND_START:R_HAND_END] # (7,)

        #  臂部控制（带平滑）
        try:
            current_arm_q = self.arm_controller.get_current_dual_arm_q()
            velocity_limit = self.arm_controller.arm_velocity_limit
            control_dt = self.arm_controller.control_dt
            delta = q_arm - current_arm_q
            motion_scale = np.max(np.abs(delta)) / (velocity_limit * control_dt)
            cliped_arm_q = current_arm_q + delta / max(motion_scale, 1.0)
            self.arm_controller.ctrl_dual_arm(cliped_arm_q, np.zeros_like(cliped_arm_q))
        except Exception as e:
            logger.error(f" Arm control failed: {e}")

        #  手部控制（DexControl）
        if self.dc is not None:
            try:
                self.dc.control_process(q_lhand, q_rhand)
                logger.debug(" Hand control sent.")
            except Exception as e:
                logger.error(f" Hand control failed: {e}")
        else:
            logger.warning("  DexControl not initialized — skipping hand control.")

        logger.info(" _set_joint_state: Arm + hand commands dispatched.")

    def _get_ee_state(self):
        """
        获取当前末端执行器状态（12维）。
        
        Returns:
            np.ndarray: shape (12,)
                [L_x, L_y, L_z, L_roll, L_pitch, L_yaw,
                R_x, R_y, R_z, R_roll, R_pitch, R_yaw]
            单位: 位置 -> "m", 旋转 -> "radian"
        """
        # 获取当前关节状态 (14D)
        q, _ = self._get_joint_state()
        
        # 调用 IK 求解器的 FK 计算末端位姿 (4x4 齐次矩阵)
        L_pose, R_pose = self.arm_ik_fk.forward_kinematics(q)
        
        # 提取位置 (3,)
        L_pos = L_pose[:3, 3]
        R_pos = R_pose[:3, 3]
        
        # 提取旋转 → 转为欧拉角 (roll, pitch, yaw) in radians
        L_rpy = pin.rpy.matrixToRpy(L_pose[:3, :3])  # shape (3,) → [roll, pitch, yaw]
        R_rpy = pin.rpy.matrixToRpy(R_pose[:3, :3])
        
        # 拼接成 12D 向量
        ee_state = np.concatenate([L_pos, L_rpy, R_pos, R_rpy])
        return ee_state


    def _set_ee_state(self, ee_state: np.ndarray):
        """
        设置末端执行器目标位姿。
        
        Args:
            ee_state (np.ndarray): shape (12,)
                [L_x, L_y, L_z, L_roll, L_pitch, L_yaw,
                R_x, R_y, R_z, R_roll, R_pitch, R_yaw]
                单位: 位置 -> meters, 旋转 -> radians (RPY, xyz order)
        """
        if len(ee_state) != 12:
            raise ValueError(f"Expected ee_state of length 12, got {len(ee_state)}")

        # --- 解析左手目标 ---
        L_pos = ee_state[0:3]
        L_rpy = ee_state[3:6]
        if self._is_rpy_singular(L_rpy):
            logger.info(f"[L] RPY near singularity (pitch={np.degrees(L_rpy[1]):.1f}°)")

        L_rot = pin.rpy.rpyToMatrix(L_rpy[0], L_rpy[1], L_rpy[2])
        L_tf_se3 = pin.SE3(L_rot, L_pos)
        L_tf = L_tf_se3.homogeneous  # 👈 转为 (4, 4) numpy.ndarray

        # --- 解析右手目标 ---
        R_pos = ee_state[6:9]
        R_rpy = ee_state[9:12]
        if self._is_rpy_singular(R_rpy):
            logger.info(f"[R] RPY near singularity (pitch={np.degrees(R_rpy[1]):.1f}°)")

        R_rot = pin.rpy.rpyToMatrix(R_rpy[0], R_rpy[1], R_rpy[2])
        R_tf_se3 = pin.SE3(R_rot, R_pos)
        R_tf = R_tf_se3.homogeneous  # 👈 转为 (4, 4) numpy.ndarray

        # --- 获取当前状态 ---
        current_q = self.arm_controller.get_current_dual_arm_q()
        current_dq = self.arm_controller.get_current_dual_arm_dq()
        
        try:
            sol_q, sol_tauff = self.arm_ik_fk.solve_ik(
                left_wrist=L_tf,    
                right_wrist=R_tf,    
                current_lr_arm_motor_q=current_q,
                current_lr_arm_motor_dq=current_dq  
            )
            self.arm_controller.ctrl_dual_arm(sol_q, sol_tauff)
        except Exception as e:
            logger.error(
                f"[_set_ee_state] IK failed for L{L_pos} R{R_pos}:\n"
                f"  L_rpy={np.degrees(L_rpy)}°, R_rpy={np.degrees(R_rpy)}°\n"
                f"  Error: {e}"
            )
            raise

    def _is_rpy_singular(self, rpy):
        pitch = rpy[1]
        return abs(abs(pitch) - np.pi / 2) < 0.05
