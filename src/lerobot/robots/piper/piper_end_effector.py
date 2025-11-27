"""
Piper end-effector robot class implementation.
"""

from .piper import Piper
from .configuration_piper import PiperEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class PiperEndEffector(Piper, BaseRobotEndEffector):
    """
    Piper robot class implementation with end effector.
    Params:
    - config: PiperEndEffectorConfig
    """
    def __init__(self, config: PiperEndEffectorConfig) -> None:
        super().__init__(config)