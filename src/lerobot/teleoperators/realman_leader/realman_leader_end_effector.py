"""
Realman end-effector leader class implementation.
"""

from .realman_leader import RealmanLeader
from .config_realman_leader import RealmanLeaderEndEffectorConfig
from ..base_leader import BaseLeaderEndEffector


class RealmanLeaderEndEffector(RealmanLeader, BaseLeaderEndEffector):
    """
    Realman robot leader implementation with end effector.
    Params:
    - config: RealmanLeaderEndEffectorConfig
    """

    config_class = RealmanLeaderEndEffectorConfig
    name = "realman_leader_end_effector"
    
    def __init__(self, config: RealmanLeaderEndEffectorConfig) -> None:
        super().__init__(config)