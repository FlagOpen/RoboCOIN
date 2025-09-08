from ..bi_base_robot import BiBaseRobot
from .configuration_bi_realman import BiRealmanConfig

from ..realman import Realman


class BiRealman(BiBaseRobot):

    config_class = BiRealmanConfig
    name = "bi_realman"

    def __init__(self, config: BiRealmanConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_robot_configs(self):
        left_config, right_config = super()._prepare_robot_configs()
        left_config.ip = self.config.ip_left
        left_config.port = self.config.port_left
        right_config.ip = self.config.ip_right
        right_config.port = self.config.port_right
        return left_config, right_config
    
    def _prepare_robots(self, left_config, right_config):
        self.left_robot = Realman(left_config)
        self.right_robot = Realman(right_config)