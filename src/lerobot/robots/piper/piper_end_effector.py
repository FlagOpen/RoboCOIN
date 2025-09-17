from .piper import Piper
from .configuration_piper import PiperEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class PiperEndEffector(Piper, BaseRobotEndEffector):
    def __init__(self, config: PiperEndEffectorConfig):
        super().__init__(config)