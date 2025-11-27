"""
BiRealman robot end effector implementation.
"""

from .configuration_bi_realman import BiRealmanEndEffectorConfig
from .bi_realman import BiRealman
from ..bi_base_robot import BiBaseRobotEndEffector
from ..realman import RealmanEndEffector, RealmanEndEffectorConfig


class BiRealmanEndEffector(BiRealman, BiBaseRobotEndEffector):
    """
    BiRealman robot end effector implementation.
    Params:
    - config: BiRealmanEndEffectorConfig
    """

    config_class = BiRealmanEndEffectorConfig
    name = "bi_realman_end_effector"

    def __init__(self, config: BiRealmanEndEffectorConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_robots(self):
        """
        Prepare the left and right Realman end effector robots.
        Initializes two RealmanEndEffector instances with respective configurations.
        """
        left_config = RealmanEndEffectorConfig(
            ip=self.config.ip_left,
            port=self.config.port_left,
            block=self.config.block,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            base_euler=self.config.base_euler,
            model_pose_units=self.config.model_pose_units,
            id=f"{self.config.id}_left" if self.config.id else None,
            cameras={},
        )
        right_config = RealmanEndEffectorConfig(
            ip=self.config.ip_right,
            port=self.config.port_right,
            block=self.config.block,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            base_euler=self.config.base_euler,
            model_pose_units=self.config.model_pose_units,
            id=f"{self.config.id}_right" if self.config.id else None,
            cameras={},
        )
        self.left_robot = RealmanEndEffector(left_config)
        self.right_robot = RealmanEndEffector(right_config)