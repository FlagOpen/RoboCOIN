"""
Bi-base leader teleoperator for end-effector control.
"""

from typing import Any, Dict
from .config_bi_base_leader import BiBaseLeaderEndEffectorConfig
from .bi_base_leader import BiBaseLeader


class BiBaseLeaderEndEffector(BiBaseLeader):
    """
    Dual-arm teleoperator leader focusing on end-effector control.
    Manages two independent teleoperator leaders (left and right) with end-effector level actions.
    Delegates operations to individual leader controllers while providing unified interface.
    Params:
    - config: BiBaseLeaderEndEffectorConfig
    e.g.
    ```python
    from lerobot.teleoperators.bi_base_leader import BiBaseLeaderEndEffector
    from lerobot.teleoperators.bi_base_leader.config_bi_base_leader import BiBaseLeaderEndEffectorConfig

    config = BiBaseLeaderEndEffectorConfig(...)
    teleop = BiBaseLeaderEndEffector(config)
    teleop.connect()
    action = teleop.get_action() # e.g. {'left_x': val1, ..., 'right_gripper': valN}
    teleop.disconnect()
    ```
    """

    config_class = BiBaseLeaderEndEffectorConfig
    name = "bi_base_leader_end_effector"

    def __init__(self, config: BiBaseLeaderEndEffectorConfig) -> None:
        """
        Initialize the dual-arm teleoperator leader for end-effector control.
        """
        super().__init__(config)
        self.config = config
    
    def _prepare_leaders(self) -> None:
        """
        Initialize left and right teleoperator leaders for end-effector control.
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