"""
Dummy leader teleoperator implementation.
"""

import numpy as np

from ..base_leader import BaseLeader
from .config_dummy_leader import DummyLeaderConfig


class DummyLeader(BaseLeader):
    """
    Dummy leader teleoperator implementation.
    Params:
    - config: DummyLeaderConfig
    """

    config_class = DummyLeaderConfig
    name = "dummy_leader"

    def __init__(self, config: DummyLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self._is_connected = False
    
    def _check_dependency(self):
        """
        Dummy leader has no external dependencies.
        """
        pass

    def _connect_arm(self) -> None:
        """
        Dummy leader has no physical arm to connect.
        """
        self._is_connected = True

    def _disconnect_arm(self) -> None:
        """
        Dummy leader has no physical arm to disconnect.
        """
        self._is_connected = False
    
    @property
    def is_connected(self) -> bool:
        """
        Check if the dummy leader is "connected".
        Returns:
        - is_connected: bool
        """
        return self._is_connected
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get the current joint state of the dummy leader.
        Returns:
        - joint_state: np.ndarray
        """
        return self.config.joint_state
    
    def _get_ee_state(self) -> np.ndarray:
        """
        Get the current end effector state of the dummy leader.
        Returns:
        - ee_state: np.ndarray
        """
        return self.config.ee_state