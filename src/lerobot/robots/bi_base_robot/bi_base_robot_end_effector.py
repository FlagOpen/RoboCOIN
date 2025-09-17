from typing import Any
from .configuration_bi_base_robot import BiBaseRobotEndEffectorConfig
from ..base_robot import BaseRobotEndEffector
from .bi_base_robot import BiBaseRobot


class BiBaseRobotEndEffector(BiBaseRobot):

    config_class = BiBaseRobotEndEffectorConfig
    name = "bi_base_robot_end_effector"

    def __init__(self, config: BiBaseRobotEndEffectorConfig):
        super().__init__(config)
        self.config = config

    def _prepare_robots(self):
        raise NotImplementedError

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            each: float for each in [
                'left_x', 'left_y', 'left_z', 'left_roll', 'left_pitch', 'left_yaw', 'left_gripper',
                'right_x', 'right_y', 'right_z', 'right_roll', 'right_pitch', 'right_yaw', 'right_gripper'
            ]
        }