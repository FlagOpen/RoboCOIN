"""
Bi-Realman leader teleoperator with end effector implementation.
"""

from .config_bi_realman_leader import BiRealmanLeaderEndEffectorConfig
from .bi_realman_leader import BiRealmanLeader
from ..bi_base_leader import BiBaseLeaderEndEffector
from ..realman_leader import RealmanLeaderEndEffector, RealmanLeaderEndEffectorConfig


class BiRealmanLeaderEndEffector(BiRealmanLeader, BiBaseLeaderEndEffector):
    """
    Bi-Realman leader teleoperator implementation with end effectors.
    Params:
    - config: BiRealmanLeaderEndEffectorConfig
    """

    config_class = BiRealmanLeaderEndEffectorConfig
    name = "bi_realman_leader_end_effector"

    def __init__(self, config: BiRealmanLeaderEndEffectorConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_leaders(self):
        """
        Prepare the left and right RealmanLeaderEndEffector leaders.
        """
        left_config = RealmanLeaderEndEffectorConfig(
            ip=self.config.ip_left,
            port=self.config.port_left,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            id=f"{self.config.id}_left" if self.config.id else None,
        )
        right_config = RealmanLeaderEndEffectorConfig(
            ip=self.config.ip_right,
            port=self.config.port_right,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            id=f"{self.config.id}_right" if self.config.id else None,
        )
        self.left_leader = RealmanLeaderEndEffector(left_config)
        self.right_leader = RealmanLeaderEndEffector(right_config)