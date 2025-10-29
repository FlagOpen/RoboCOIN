"""
Configuration for Dummy robot.
"""
import numpy as np

from ..base_robot import BaseRobot
from .configuration_dummy import DummyRobotConfig


class DummyRobot(BaseRobot):
    """
    Dummy robot implementation.
    Params:
    - config: DummyRobotConfig
    """

    config_class = DummyRobotConfig
    name = "dummy"

    def __init__(self, config: DummyRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self._joint_state = np.zeros(len(self.config.joint_names))
        self._ee_state = np.zeros(7)  # Assuming 7 DOF

    def _check_dependency(self):
        """
        Check for dependencies required by the dummy robot.
        Dummy implementation does nothing.
        """
        pass
    
    def _connect_arm(self):
        """
        Connect to the dummy robot arm.
        Dummy implementation does nothing.
        """
        pass
    
    def _disconnect_arm(self):
        """
        Disconnect from the dummy robot arm.
        Dummy implementation does nothing.
        """
        pass
    
    def _set_joint_state(self, state: np.ndarray):
        """
        Set the joint state of the dummy robot.
        Dummy implementation just stores the state.
        """
        self._joint_state = state
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get the joint state of the dummy robot.
        Dummy implementation just returns the stored state.
        """
        return self._joint_state

    def _set_ee_state(self, state: np.ndarray):
        """
        Set the end-effector state of the dummy robot.
        Dummy implementation just stores the state.
        """
        self._ee_state = state

    def _get_ee_state(self) -> np.ndarray:
        """
        Get the end-effector state of the dummy robot.
        Dummy implementation just returns the stored state.
        """
        return self._ee_state