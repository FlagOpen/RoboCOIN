"""
Bi-Piper leader teleoperator implementation.
"""

from ..bi_base_leader import BiBaseLeader
from .config_bi_piper_leader import BiPiperLeaderConfig

from ..piper_leader import PiperLeader, PiperLeaderConfig


class BiPiperLeader(BiBaseLeader):
    """
    Bi-Piper leader teleoperator implementation.
    Params:
    - config: BiPiperLeaderConfig
    """

    config_class = BiPiperLeaderConfig
    name = "bi_piper_leader"

    def __init__(self, config: BiPiperLeaderConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_leaders(self):
        """
        Prepare the left and right PiperLeader leaders.
        """
        left_config = PiperLeaderConfig(
            can=self.config.can_left,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_left" if self.config.id else None,
        )
        right_config = PiperLeaderConfig(
            can=self.config.can_right,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_right" if self.config.id else None,
        )
        self.left_leader = PiperLeader(left_config)
        self.right_leader = PiperLeader(right_config)