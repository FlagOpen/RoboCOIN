from ..bi_base_robot import BiBaseRobot
from .configuration_bi_realman import BiRealmanConfig

from ..realman import Realman, RealmanConfig


class BiRealman(BiBaseRobot):

    config_class = BiRealmanConfig
    name = "bi_realman"

    def __init__(self, config: BiRealmanConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_robot_configs(self):
        left_config = RealmanConfig(
            ip=self.config.ip_left,
            port=self.config.port_left,
            block=self.config.block,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            model_units=self.config.model_units,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            delta_with=self.config.delta_with,
            visualize=False,
            id=f"{self.config.id}_left" if self.config.id else None,
            cameras={},
        )
        right_config = RealmanConfig(
            ip=self.config.ip_right,
            port=self.config.port_right,
            block=self.config.block,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            model_units=self.config.model_units,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            delta_with=self.config.delta_with,
            visualize=False,
            id=f"{self.config.id}_right" if self.config.id else None,
            cameras={},
        )

        return left_config, right_config
    
    def _prepare_robots(self, left_config, right_config):
        self.left_robot = Realman(left_config)
        self.right_robot = Realman(right_config)