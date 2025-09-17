from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from ..bi_base_robot import BiBaseRobotConfig, BiBaseRobotEndEffectorConfig


@RobotConfig.register_subclass("bi_realman")
@dataclass
class BiRealmanConfig(BiBaseRobotConfig):
    ip_left: str = "169.254.128.18"
    port_left: int = 8080
    ip_right: str = "169.254.128.19"
    port_right: int = 8080
    block: bool = False
    wait_second: float = 0.1
    velocity: int = 30
    
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])
    
    init_state_left: list[float] = field(default_factory=lambda: [
        -0.84, -2.03,  1.15,  1.15,  2.71,  1.60, -2.99, 888.00,
    ])
    init_state_right: list[float] = field(default_factory=lambda: [
         1.16,  2.01, -0.79, -0.68, -2.84, -1.61,  2.37, 832.00,
    ])


@RobotConfig.register_subclass("bi_realman_end_effector")
@dataclass
class BiRealmanEndEffectorConfig(BiRealmanConfig, BiBaseRobotEndEffectorConfig):
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])