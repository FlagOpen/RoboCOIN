"""
Bi-Dummy leader teleoperator implementation with end effectors.
"""

from .config_bi_dummy_leader import BiDummyLeaderEndEffectorConfig
from .bi_dummy_leader import BiDummyLeader
from ..bi_base_leader import BiBaseLeaderEndEffector
from ..dummy_leader import DummyLeaderEndEffector, DummyLeaderEndEffectorConfig


class BiDummyLeaderEndEffector(BiDummyLeader, BiBaseLeaderEndEffector):
    """
    Bi-Dummy leader teleoperator implementation with end effectors.
    Params:
    - config: BiDummyLeaderEndEffectorConfig
    """

    config_class = BiDummyLeaderEndEffectorConfig
    name = "bi_dummy_leader_end_effector"

    def __init__(self, config: BiDummyLeaderEndEffectorConfig):
        super().__init__(config)
        self.config = config

    def _prepare_leaders(self):
        """
        Prepare the left and right DummyLeaderEndEffector leaders.
        """
        left_config = DummyLeaderEndEffectorConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_left" if self.config.id else None,
        )
        right_config = DummyLeaderEndEffectorConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            id=f"{self.config.id}_right" if self.config.id else None,
        )
        self.left_leader = DummyLeaderEndEffector(left_config)
        self.right_leader = DummyLeaderEndEffector(right_config)