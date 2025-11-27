"""
Bi-Base Robot End Effector Module
"""

from typing import Any, Dict
from .configuration_bi_base_robot import BiBaseRobotEndEffectorConfig
from .bi_base_robot import BiBaseRobot


class BiBaseRobotEndEffector(BiBaseRobot):
    """
    Dual-arm robot system focusing on end-effector control.
    Manages two independent robot arms (left and right) with end-effector level actions.
    Delegates operations to individual arm controllers while providing unified interface.
    Params:
    - config: BiBaseRobotEndEffectorConfig
    e.g.
    ```python
    from lerobot.robots.bi_base_robot import BiBaseRobotEndEffector
    from lerobot.robots.bi_base_robot.configuration_bi_base_robot import BiBaseRobotEndEffectorConfig

    config = BiBaseRobotEndEffectorConfig(...)
    robot = BiBaseRobotEndEffector(config)
    robot.connect()
    obs = robot.get_observation()
    action = {...}  # Define end-effector actions for both arms
    robot.send_action(action)
    robot.disconnect()
    ```
    """

    config_class = BiBaseRobotEndEffectorConfig
    name = "bi_base_robot_end_effector"

    def __init__(self, config: BiBaseRobotEndEffectorConfig) -> None:
        """
        Initialize the dual-arm end-effector robot system.
        """
        super().__init__(config)
        self.config = config

    def _prepare_robots(self) -> None:
        """
        Initialize left and right robot arm controllers for end-effector control.
        """
        raise NotImplementedError

    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Define action features for both arms at end-effector level.
        Returns:
        - Dictionary with action feature names and their types.
        """
        return {
            each: float for each in [
                'left_x', 'left_y', 'left_z', 'left_roll', 'left_pitch', 'left_yaw', 'left_gripper',
                'right_x', 'right_y', 'right_z', 'right_roll', 'right_pitch', 'right_yaw', 'right_gripper'
            ]
        }