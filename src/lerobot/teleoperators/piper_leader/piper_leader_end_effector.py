"""
Piper end-effector leader class implementation.
"""

from .piper_leader import PiperLeader
from .config_piper_leader import PiperLeaderEndEffectorConfig
from ..base_leader import BaseLeaderEndEffector


class PiperLeaderEndEffector(PiperLeader, BaseLeaderEndEffector):
    """
    Piper robot leader implementation with end effector.
    Params:
    - config: PiperLeaderEndEffectorConfig
    """

    config_class = PiperLeaderEndEffectorConfig
    name = "piper_leader_end_effector"

    def __init__(self, config: PiperLeaderEndEffectorConfig) -> None:
        super().__init__(config)