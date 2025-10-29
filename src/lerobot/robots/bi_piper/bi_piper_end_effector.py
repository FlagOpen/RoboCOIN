"""
BiPiper end-effector robot class implementation.
"""

from .configuration_bi_piper import BiPiperEndEffectorConfig
from .bi_piper import BiPiper
from ..bi_base_robot import BiBaseRobotEndEffector
from ..piper import PiperEndEffector, PiperEndEffectorConfig


class BiPiperEndEffector(BiPiper, BiBaseRobotEndEffector):
    """
    BiPiper robot class implementation with end effectors.
    Params:
    - config: BiPiperEndEffectorConfig
    """

    config_class = BiPiperEndEffectorConfig
    name = "bi_piper_end_effector"

    def __init__(self, config: BiPiperEndEffectorConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_robots(self):
        """
        Prepare the left and right PiperEndEffector robots.
        Initializes two PiperEndEffector instances with appropriate configurations.
        """
        left_config = PiperEndEffectorConfig(
            can=self.config.can_left,
            velocity=self.config.velocity,
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
        right_config = PiperEndEffectorConfig(
            can=self.config.can_right,
            velocity=self.config.velocity,
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
        self.left_robot = PiperEndEffector(left_config)
        self.right_robot = PiperEndEffector(right_config)