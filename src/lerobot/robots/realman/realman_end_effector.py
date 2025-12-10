"""
Realman robot end effector implementation.
"""

from .realman import Realman
from .configuration_realman import RealmanEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class RealmanEndEffector(Realman, BaseRobotEndEffector):
    """
    Realman robot end effector implementation.
    Params:
    - config: RealmanEndEffectorConfig
    """

    config_class = RealmanEndEffectorConfig
    name = "realman_end_effector"
    
    def __init__(self, config: RealmanEndEffectorConfig) -> None:
        super().__init__(config)