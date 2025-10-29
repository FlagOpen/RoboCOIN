"""
Dummy robot end effector implementation.
"""

from .dummy_robot import DummyRobot
from .configuration_dummy import DummyRobotEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class DummyRobotEndEffector(DummyRobot, BaseRobotEndEffector):
    """
    Dummy robot end effector implementation.
    Params:
    - config: DummyRobotEndEffectorConfig
    """

    config_class = DummyRobotEndEffectorConfig
    name = "dummy_end_effector"

    def __init__(self, config: DummyRobotEndEffectorConfig) -> None:
        super().__init__(config)