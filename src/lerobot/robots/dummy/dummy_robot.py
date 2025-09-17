from ..base_robot import BaseRobot
from .configuration_dummy import DummyRobotConfig


class DummyRobot(BaseRobot):

    config_class = DummyRobotConfig
    name = "dummy"

    def __init__(self, config: DummyRobotConfig):
        super().__init__(config)
        self.config = config
        self._joint_state = [0] * len(self.config.joint_names)
        self._ee_state = [0] * 7  # Assuming 7 DOF

    def _check_dependency(self):
        pass
    
    def _connect_arm(self):
        pass
    
    def _disconnect_arm(self):
        pass
    
    def _set_joint_state(self, state: list[int]):
        self._joint_state = state
    
    def _get_joint_state(self) -> list[int]:
        return self._joint_state

    def _set_ee_state(self, state: list[float]):
        self._ee_state = state

    def _get_ee_state(self) -> list[float]:
        return self._ee_state