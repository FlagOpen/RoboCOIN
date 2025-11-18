"""
Moveit end-effector robot class implementation.
"""

from .moveit_robot import MoveitRobot
from .configuration_moveit_robot import MoveitRobotEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class MoveitRobotEndEffector(MoveitRobot, BaseRobotEndEffector):
    """
    Moveit robot class implementation with end effector.
    Params:
    - config: MoveitRobotEndEffectorConfig
    """
    def __init__(self, config: MoveitRobotEndEffectorConfig) -> None:
        super().__init__(config)