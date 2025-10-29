"""
Bi-Dummy robot implementation.
"""

from ..bi_base_robot import BiBaseRobot
from .configuration_bi_dummy import BiDummyRobotConfig

from ..dummy import DummyRobot, DummyRobotConfig


class BiDummyRobot(BiBaseRobot):
    """
    Bi-Dummy robot implementation.
    Params:
    - config: BiDummyRobotConfig
    """
    
    config_class = BiDummyRobotConfig
    name = "bi_dummy"

    def __init__(self, config: BiDummyRobotConfig) -> None:
        super().__init__(config)
        self.config = config
    
    def _prepare_robots(self) -> None:
        """
        Prepare left and right DummyRobot instances.
        """
        left_config = DummyRobotConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            id=f"{self.config.id}_left" if self.config.id else None,
            cameras={},
        )
        right_config = DummyRobotConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            id=f"{self.config.id}_right" if self.config.id else None,
            cameras={},
        )
        self.left_robot = DummyRobot(left_config)
        self.right_robot = DummyRobot(right_config)