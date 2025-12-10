"""
Bi-Realman leader teleoperator implementation.
"""

from ..bi_base_leader import BiBaseLeader
from .config_bi_realman_leader import BiRealmanLeaderConfig

from ..realman_leader import RealmanLeader, RealmanLeaderConfig


class BiRealmanLeader(BiBaseLeader):
    """
    Bi-Realman leader teleoperator implementation.
    Params:
    - config: BiRealmanLeaderConfig
    """

    config_class = BiRealmanLeaderConfig
    name = "bi_realman_leader"

    def __init__(self, config: BiRealmanLeaderConfig):
        super().__init__(config)
        self.config = config

    def _prepare_leaders(self):
        """
        Prepare the left and right RealmanLeader leaders.
        """
        left_config = RealmanLeaderConfig(
            ip=self.config.ip_left,
            port=self.config.port_left,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            id=f"{self.config.id}_left" if self.config.id else None,
        )
        right_config = RealmanLeaderConfig(
            ip=self.config.ip_right,
            port=self.config.port_right,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            id=f"{self.config.id}_right" if self.config.id else None,
        )
        self.left_leader = RealmanLeader(left_config)
        self.right_leader = RealmanLeader(right_config)