"""
Bi-Piper Leader Teleoperator Module with End Effector Implementation.
"""

from .config_bi_piper_leader import BiPiperLeaderEndEffectorConfig
from .bi_piper_leader import BiPiperLeader
from ..bi_base_leader import BiBaseLeaderEndEffector
from ..piper_leader import PiperLeaderEndEffector, PiperLeaderEndEffectorConfig


class BiPiperLeaderEndEffector(BiPiperLeader, BiBaseLeaderEndEffector):
    """
    Bi-Piper leader teleoperator implementation with end effectors.
    Params:
    - config: BiPiperLeaderEndEffectorConfig
    """

    config_class = BiPiperLeaderEndEffectorConfig
    name = "bi_piper_leader_end_effector"

    def __init__(self, config: BiPiperLeaderEndEffectorConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_leaders(self):
        """
        Prepare the left and right PiperLeaderEndEffector leaders.
        """
        left_config = PiperLeaderEndEffectorConfig(
            can=self.config.can_left,
            velocity=self.config.velocity,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_left" if self.config.id else None,
        )
        right_config = PiperLeaderEndEffectorConfig(
            can=self.config.can_right,
            velocity=self.config.velocity,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_right" if self.config.id else None,
        )
        self.left_leader = PiperLeaderEndEffector(left_config)
        self.right_leader = PiperLeaderEndEffector(right_config)