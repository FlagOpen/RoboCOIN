"""
Dummy end-effector leader class implementation.
"""

from .dummy_leader import DummyLeader
from .config_dummy_leader import DummyLeaderEndEffectorConfig
from ..base_leader import BaseLeaderEndEffector


class DummyLeaderEndEffector(DummyLeader, BaseLeaderEndEffector):
    """
    Dummy robot leader implementation with end effector.
    Params:
    - config: DummyLeaderEndEffectorConfig
    """

    config_class = DummyLeaderEndEffectorConfig
    name = "dummy_leader_end_effector"

    def __init__(self, config: DummyLeaderEndEffectorConfig) -> None:
        super().__init__(config)