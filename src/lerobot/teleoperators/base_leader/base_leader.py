"""
Base leader class with joint control.
"""

import numpy as np
from typing import Any, Dict

from lerobot.errors import DeviceNotConnectedError

from .config_base_leader import BaseLeaderConfig
from .units_transform import UnitsTransform
from ..teleoperator import Teleoperator


class BaseLeader(Teleoperator):
    """
    Base leader class with joint control.
    Subclasses should implement hardware-specific communication methods.
    Supports:
    1. Joint & End-Effector control
    2. Unified unit management
    Params:
    - config: BaseLeaderConfig
    e.g.
    ```python
    from lerobot.teleoperators.base_leader import BaseLeader, BaseLeaderConfig

    config = BaseLeaderConfig(
        joint_names=['joint1_pos', 'joint2_pos', 'joint3_pos', 'gripper'],
        joint_units=['radian', 'radian', 'radian', 'meter'],
        model_joint_units=['radian', 'radian', 'radian', 'meter'],
    )
    teleop = BaseLeader(config)
    teleop.connect()
    teleop.get_action() # return action features, e.g. joint1_pos, joint2_pos, joint3_pos, gripper
    teleop.disconnect()
    ```
    """

    config_class = BaseLeaderConfig
    name = "base_leader"

    def __init__(self, config: BaseLeaderConfig) -> None:
        """Initialize the teleoperator with configuration settings"""

        super().__init__(config)
        self._check_dependency()

        self.config = config
        self.arm = None
        
        self.joint_transform = UnitsTransform(config.joint_units)
        self.pose_transform = UnitsTransform(config.pose_units)
        self.model_joint_transform = UnitsTransform(config.model_joint_units)

    def _check_dependency(self) -> None:
        """
        Check for required dependencies and libraries.
        Should be implemented by subclasses 
        to verify necessary hardware libraries are available.
        """
        return
    
    def _connect_arm(self):
        """
        Establish connection to the robot arm hardware.
        This method must be implemented by subclasses 
        to handle hardware-specific connection logic.
        """
        raise NotImplementedError
    
    def _disconnect_arm(self):
        """
        Disconnect from the robot arm hardware.
        This method must be implemented by subclasses 
        to handle hardware-specific disconnection logic.
        """
        raise NotImplementedError
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get joint positions from hardware.
        This method must be implemented by subclasses 
        to retrieve joint states from the physical robot.
        Returns:
        - state: Joint positions in robot-specific units
        """
        raise NotImplementedError
    
    def _get_ee_state(self) -> np.ndarray:
        """
        Get end-effector pose from hardware.
        This method must be implemented by subclasses
        to retrieve end-effector states from the physical robot.
        Returns:
        - state: End-effector pose in robot-specific units
        """
        raise NotImplementedError

    def get_joint_state(self) -> np.ndarray:
        """
        Get joint positions with automatic unit conversion.
        Retrieves joint positions from hardware and converts from robot-specific units to standard units.
        Returns:
        - state: Joint positions in standard units
        """
        state = self._get_joint_state()
        return self.joint_transform.input_transform(state) # joint -> standard
    
    def get_ee_state(self) -> np.ndarray:
        """
        Get end-effector pose with automatic unit conversion.
        Retrieves end-effector pose from hardware and converts from robot-specific units to standard units.
        Returns:
        - state: End-effector pose in standard units
        """
        state = self._get_ee_state()
        return self.pose_transform.input_transform(state) # end_effector -> standard
    
    def connect(self) -> None:
        """
        Connect to the robot and initialize components.
        """
        self._connect_arm()

    def disconnect(self) -> None:
        """
        Disconnect from the robot and clean up resources.
        """
        self._disconnect_arm()
    
    def check_initialized(self) -> bool:
        """
        Check if the robot is initialized based on configuration.
        Returns:
        - bool indicating if the robot is initialized
        """
        if self.config.init_type == "none":
            return True
        
        target_state = np.array(self.config.init_state)
        if self.config.init_type == "joint":
            current_state = self.get_joint_state()
        elif self.config.init_type == "end_effector":
            current_state = self.get_ee_state()
        
        for i in range(len(target_state)):
            if abs(current_state[i] - target_state[i]) > self.config.init_threshold:
                print(f"{self.id}: Joint {i+1} not initialized: target {target_state[i]}, current {current_state[i]}")
                return False
        
        return True
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get current action with unit conversion.
        Returns:
        - action_dict: Dictionary containing joint positions and gripper state in model units
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state = self.get_joint_state()
        state_to_send = self.model_joint_transform.output_transform(state) # standard -> model
        action_dict = {k: v for k, v in zip(self._motors_ft.keys(), state_to_send)}
        return action_dict
    
    @property
    def _motors_ft(self) -> Dict[str, Any]:
        """
        Motor joint features dictionary.
        Returns:
        - dict mapping joint names to float types
        """
        return {
            f'{each}_pos': float for each in self.config.joint_names
        }

    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Action features dictionary.
        Returns:
        - dict mapping joint names to float types
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot and all cameras are connected.
        Returns:
        - bool indicating connection status
        """
        raise NotImplementedError
    
    def is_calibrated(self) -> bool:
        """
        Check if the robot is calibrated.
        Returns:
        - bool indicating calibration status, True by default
        """
        return True
    
    def calibrate(self) -> None:
        """
        Calibrate the robot, doing nothing by default.
        """
        pass

    def configure(self) -> None:
        """
        Configure the robot, doing nothing by default.
        """
        pass
    
    @property
    def feedback_features(self) -> Dict[str, Any]:
        """
        Feedback features dictionary.
        Returns:
        - dict mapping feedback names to float types
        """
        return {}
    
    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError