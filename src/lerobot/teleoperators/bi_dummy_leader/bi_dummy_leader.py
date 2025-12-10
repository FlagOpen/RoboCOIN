"""
Bi-Dummy leader teleoperator implementation.
"""

from ..bi_base_leader import BiBaseLeader
from .config_bi_dummy_leader import BiDummyLeaderConfig

from ..dummy_leader import DummyLeader, DummyLeaderConfig


class BiDummyLeader(BiBaseLeader):
    """
    Bi-Dummy leader teleoperator implementation.
    Params:
    - config: BiDummyLeaderConfig
    """

    config_class = BiDummyLeaderConfig
    name = "bi_dummy_leader"

    def __init__(self, config: BiDummyLeaderConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_leaders(self):
        """
        Prepare the left and right DummyLeader leaders.
        """
        left_config = DummyLeaderConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_left" if self.config.id else None,
        )
        right_config = DummyLeaderConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_right" if self.config.id else None,
        )
        self.left_leader = DummyLeader(left_config)
        self.right_leader = DummyLeader(right_config)