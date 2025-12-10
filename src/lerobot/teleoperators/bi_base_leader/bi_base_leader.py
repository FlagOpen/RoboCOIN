"""
Bi-base teleoperator leader for dual-arm robotic systems.
"""

import numpy as np
from typing import Any, Dict

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_bi_base_leader import BiBaseLeaderConfig


class BiBaseLeader(Teleoperator):
    """
    Dual-arm teleoperator leader.
    Manages two independent teleoperator leaders (left and right).
    Delegates operations to individual leader controllers while providing unified interface.
    Params:
    - config: BiBaseLeaderConfig
    e.g.
    ```python
    from lerobot.teleoperators.bi_base_leader import BiBaseLeader
    from lerobot.teleoperators.bi_base_leader.config_bi_base_leader import BiBaseLeaderConfig

    config = BiBaseLeaderConfig(...)
    teleop = BiBaseLeader(config)
    teleop.connect()
    action = teleop.get_action() # e.g. {'left_joint1': val1, ..., 'right_jointN': valN}
    teleop.disconnect()
    ```
    """

    config_class = BiBaseLeaderConfig
    name = "bi_base_leader"

    def __init__(self, config: BiBaseLeaderConfig) -> None:
        """
        Initialize the dual-arm teleoperator leader.
        """
        super().__init__(config)
        self.config = config
        self._prepare_leaders()
    
    def _prepare_leaders(self) -> None:
        """
        Initialize left and right teleoperator leaders.
        This method must be implemented by subclasses to create and configure:
        - self.left_leader: Controller for left arm (instance of BaseLeader or similar)
        - self.right_leader: Controller for right arm (instance of BaseLeader or similar)
        """
        raise NotImplementedError
    
    def check_initialized(self) -> bool:
        """
        Check if both leaders are properly initialized.
        Returns:
        - True if both left and right leaders are initialized, False otherwise.
        """
        return self.left_leader.check_initialized() and self.right_leader.check_initialized()

    def get_action(self) -> Dict[str, Any]:
        """
        Get action commands from both leaders.
        Returns:
        - action: Combined action commands from left and right leaders with 'left_' and 'right_' prefixes.
        """
        action_left = self.left_leader.get_action()
        action_right = self.right_leader.get_action()

        action_left = {f"left_{k}": v for k, v in action_left.items()}
        action_right = {f"right_{k}": v for k, v in action_right.items()}
        return {**action_left, **action_right}
    
    @property
    def _motors_ft(self) -> Dict[str, Any]:
        """
        Define motor feature types for both arms.
        Combines features from left and right arms with prefixes.
        Returns:
        - motors_dict: Combined motor feature types with 'left_' and 'right_' prefixes.
        """
        left_ft = {f"left_{each}": float for each in self.left_leader._motors_ft.keys()}
        right_ft = {f"right_{each}": float for each in self.right_leader._motors_ft.keys()}
        return {**left_ft, **right_ft}
    
    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Define action features for both arms.
        Returns:
        - Dictionary with action feature names and their types.
        """
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:
        """
        Check if both leaders are connected.
        Returns:
        - True if both left and right leaders are connected, False otherwise.
        """
        return self.left_leader.is_connected and self.right_leader.is_connected
    
    def get_joint_state(self) -> np.ndarray:
        """
        Get current joint states for both arms.
        Returns:
        - state: Concatenated joint states from left and right arms.
        """
        state_left = self.left_leader.get_joint_state()
        state_right = self.right_leader.get_joint_state()
        return np.concatenate([state_left, state_right])
    
    def get_ee_state(self) -> np.ndarray:
        """
        Get current end-effector states for both arms.
        Returns:
        - state: Concatenated end-effector states from left and right arms.
        """
        state_left = self.left_robot.get_ee_state()
        state_right = self.right_robot.get_ee_state()
        return np.concatenate([state_left, state_right])
    
    def connect(self) -> None:
        """
        Connect to both leaders.
        """
        self.left_leader.connect()
        self.right_leader.connect()

    def is_calibrated(self) -> bool:
        """
        Check if both leaders are calibrated.
        Returns:
        - True if both left and right leaders are calibrated, False otherwise.
        """
        return self.left_leader.is_calibrated() and self.right_leader.is_calibrated()
    
    def calibrate(self) -> None:
        """
        Calibrate both leaders.
        """
        self.left_leader.calibrate()
        self.right_leader.calibrate()
    
    def configure(self) -> None:
        """
        Configure both leaders.
        """
        self.left_leader.configure()
        self.right_leader.configure()
    
    def disconnect(self) -> None:
        """
        Disconnect from both leaders.
        """
        self.left_leader.disconnect()
        self.right_leader.disconnect()