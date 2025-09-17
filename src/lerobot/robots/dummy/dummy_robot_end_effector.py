from .dummy_robot import DummyRobot
from .configuration_dummy import DummyRobotEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class DummyRobotEndEffector(DummyRobot, BaseRobotEndEffector):
    def __init__(self, config: DummyRobotEndEffectorConfig):
        super().__init__(config)