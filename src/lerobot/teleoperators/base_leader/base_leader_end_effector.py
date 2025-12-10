"""
Base class for teleoperators via end-effector commands.
"""

from typing import Any, Dict

from lerobot.errors import DeviceNotConnectedError

from .base_leader import BaseLeader
from .config_base_leader import BaseLeaderEndEffectorConfig
from .units_transform import UnitsTransform


class BaseLeaderEndEffector(BaseLeader):
    """
    Base class for teleoperators via end-effector (EE) commands.
    Extends the BaseLeader class to provide end-effector level control.
    Handles unit conversions and action preparation specific to EE control.
    Params:
    - config: Configuration object for the end-effector teleoperator.
    e.g.
    ```python
    from lerobot.teleoperators.base_leader import BaseLeaderEndEffector, BaseLeaderEndEffectorConfig

    config = BaseLeaderEndEffectorConfig(
        pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
        model_pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
    )
    teleop = BaseLeaderEndEffector(config)
    teleop.connect()
    teleop.get_action() # return action features, e.g. x, y, z, roll, pitch, yaw, gripper
    teleop.disconnect()
    ```
    """

    config_class = BaseLeaderEndEffectorConfig
    name = "base_leader_end_effector"

    def __init__(self, config: BaseLeaderEndEffectorConfig) -> None:
        """
        Initialize the end-effector teleoperator with configuration.
        """
        super().__init__(config)
        self.model_pose_transform = UnitsTransform(config.model_pose_units)
    
    def connect(self) -> None:
        """
        Connect to the teleoperator
        """
        super().connect()
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get current action with unit conversion for end-effector control.
        Returns:
        - action_dict: Dictionary of action features for end-effector control
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state = self.get_ee_state()
        state_to_send = self.model_joint_transform.output_transform(state) # standard -> model
        action_dict = {k: v for k, v in zip(self.action_features, state_to_send)}
        return action_dict

    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Define the action features for end-effector control.
        Returns:
        - Dictionary mapping action feature names to their types
        """
        return {
            each: float for each in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        }