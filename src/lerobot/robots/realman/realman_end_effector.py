from .realman import Realman
from .configuration_realman import RealmanEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class RealmanEndEffector(Realman, BaseRobotEndEffector):
    def __init__(self, config: RealmanEndEffectorConfig):
        super().__init__(config)