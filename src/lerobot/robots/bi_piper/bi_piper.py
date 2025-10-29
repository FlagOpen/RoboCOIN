"""
BiPiper robot class implementation.
"""

from ..bi_base_robot import BiBaseRobot
from .configuration_bi_piper import BiPiperConfig

from ..piper import Piper, PiperConfig


class BiPiper(BiBaseRobot):
    """
    BiPiper robot class implementation.
    Params:
    - config: BiPiperConfig
    """

    config_class = BiPiperConfig
    name = "bi_piper"

    def __init__(self, config: BiPiperConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_robots(self):
        """
        Prepare the left and right Piper robots.
        Initializes two Piper instances with appropriate configurations.
        """
        left_config = PiperConfig(
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
            id=f"{self.config.id}_left" if self.config.id else None,
            cameras={},
        )
        right_config = PiperConfig(
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
            id=f"{self.config.id}_right" if self.config.id else None,
            cameras={},
        )
        self.left_robot = Piper(left_config)
        self.right_robot = Piper(right_config)