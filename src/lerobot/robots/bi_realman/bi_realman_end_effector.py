from .configuration_bi_realman import BiRealmanEndEffectorConfig
from .bi_realman import BiRealman
from ..bi_base_robot import BiBaseRobotEndEffector
from ..realman import RealmanEndEffector


class BiRealmanEndEffector(BiRealman, BiBaseRobotEndEffector):

    config_class = BiRealmanEndEffectorConfig
    name = "bi_realman_end_effector"

    def __init__(self, config: BiRealmanEndEffectorConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_robots(self):
        self.left_robot = RealmanEndEffector(self.left_config)
        self.right_robot = RealmanEndEffector(self.right_config)