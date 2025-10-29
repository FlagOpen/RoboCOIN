"""
Bi-Dummy robot end effector implementation.
"""

from .configuration_bi_dummy import BiDummyRobotEndEffectorConfig
from .bi_dummy_robot import BiDummyRobot
from ..bi_base_robot import BiBaseRobotEndEffector
from ..dummy import DummyRobotEndEffector


class BiDummyRobotEndEffector(BiDummyRobot, BiBaseRobotEndEffector):
    """
    Bi-Dummy robot end effector implementation.
    Params:
    - config: BiDummyRobotEndEffectorConfig
    """
    
    config_class = BiDummyRobotEndEffectorConfig
    name = "bi_dummy_end_effector"

    def __init__(self, config: BiDummyRobotEndEffectorConfig) -> None:
        super().__init__(config)
        self.config = config
    
    def _prepare_robots(self) -> None:
        """
        Prepare left and right DummyRobotEndEffector instances.
        """
        left_config = BiDummyRobotEndEffectorConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            base_euler=self.config.base_euler,
            model_pose_units=self.config.model_pose_units,
            id=f"{self.config.id}_left" if self.config.id else None,
            cameras={},
        )
        right_config = BiDummyRobotEndEffectorConfig(
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            base_euler=self.config.base_euler,
            model_pose_units=self.config.model_pose_units,
            id=f"{self.config.id}_right" if self.config.id else None,
            cameras={},
        )
        self.left_robot = DummyRobotEndEffector(left_config)
        self.right_robot = DummyRobotEndEffector(right_config)