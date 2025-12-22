from dataclasses import dataclass, field
from typing import List, Optional
from lerobot.robots import RobotConfig

# 主控制器：控制 G1 双臂（14 DoF），提供 EE 位姿接口
from robot_control.robot_arm import G1_29_ArmController

# 手部控制器
from robot_control.robot_hand_unitree import Dex3_1_Controller


@RobotConfig.register_subclass("g1_29")
@dataclass
class G1_29_RobotConfig(RobotConfig):
    robot_type: str = "g1_29"
    robot_name: str = "Unitree G1 Dual Arms"

      # === 关节相关 ===
    joint_names: List[str] = field(default_factory=lambda: [
        # Left arm
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        # Right arm
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        # Left hand 
        "left_thumb_flex_joint", "left_index_flex_joint", "left_middle_flex_joint",
        "left_ring_flex_joint", "left_pinky_flex_joint", "left_thumb_abd_joint", "left_wrist_yaw_joint",
        # Right hand 
        "right_thumb_flex_joint", "right_index_flex_joint", "right_middle_flex_joint",
        "right_ring_flex_joint", "right_pinky_flex_joint", "right_thumb_abd_joint", "right_wrist_yaw_joint",
    ])
    num_joints: int = 14 + 14  

    # 关节控制单位：全部为弧度（Unitree SDK 要求）
    joint_units: List[str] = field(default_factory=lambda: [
        # Arm: 14 × "radian"
        "radian", "radian", "radian", "radian", "radian", "radian", "radian",
        "radian", "radian", "radian", "radian", "radian", "radian", "radian",
        #  Hand: 14 × "radian" (left 7 + right 7)
        "radian", "radian", "radian", "radian", "radian", "radian", "radian",
        "radian", "radian", "radian", "radian", "radian", "radian", "radian",
    ])


    # === 末端执行器相关 ===
    pose_units: List[str] = field(default_factory=lambda: [
        "m", "m", "m", "radian", "radian", "radian",   # left EE (6D)
        "m", "m", "m", "radian", "radian", "radian"    # right EE (6D)
    ])

    sdk_name: str = "unitree_sdk2py"
    control_frequency_hz: int = 100

