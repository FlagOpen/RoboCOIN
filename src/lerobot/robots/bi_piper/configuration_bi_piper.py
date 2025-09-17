from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_piper")
@dataclass
class BiPiperConfig(BiBaseRobotConfig):
    can_left: str = "can0"
    can_right: str = "can1"
    velocity: int = 30
    
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper',
    ])
    
    init_state_left: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 60000,
    ])
    init_state_right: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 60000,
    ])


@RobotConfig.register_subclass("bi_piper_end_effector")
@dataclass
class BiPiperEndEffectorConfig(BiPiperConfig, BiBaseRobotEndEffectorConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])