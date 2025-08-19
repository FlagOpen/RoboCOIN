"""
This module is used to convert the state representation of all robotic arms 
into some kind of generic intermediate state, and the benefits are: 

1. The actions collected on different robotic arms can be trained in a unified representation.
2. The unified action representation predicted by the model can be translated into 
   a specific representation of different robotic arms.

This practice is especially suitable for scenarios that need to support multi-ontology and control forms.
According to Pika's data format, we set all distance units to 1m, angle units to 1 radian, 
and gripper opening and closing states to 1.6 and 0, respectively.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseStandardization(ABC):
    """
    Base class for standardization of robot states and actions.
    """

    @abstractmethod
    def input_transform(self, states):
        """
        Transform the input states to a standardized format.
        """

        pass

    @abstractmethod
    def output_transform(self, states):
        """
        Transform the output actions to a standardized format.
        """

        pass


class DummyStandardization(BaseStandardization):
    """
    Dummy standardization that does not change the input states or output actions.
    """

    def input_transform(self, states):
        return states

    def output_transform(self, states):
        return states


class PiperJointStandardization(BaseStandardization):
    """
    Standardization for Piper robot joints.
    """
    def input_transform(self, states):
        return [
            states[0] * 1e-3 * np.pi / 180,  # joint_1: 0.001 degree to 1 radian
            states[1] * 1e-3 * np.pi / 180,  # joint_2: 0.001 degree to 1 radian
            states[2] * 1e-3 * np.pi / 180,  # joint_3: 0.001 degree to 1 radian
            states[3] * 1e-3 * np.pi / 180,  # joint_4: 0.001 degree to 1 radian
            states[4] * 1e-3 * np.pi / 180,  # joint_5: 0.001 degree to 1 radian
            states[5] * 1e-3 * np.pi / 180,  # joint_6: 0.001 degree to 1 radian
            states[6] / 60000.0 * 1.6  # gripper: [0, 60000] to [0, 1.6]
        ]
    
    def output_transform(self, states):
        return [
            int(states[0] * 180 / np.pi * 1e3),  # joint_1: 1 radian to 0.001 degree
            int(states[1] * 180 / np.pi * 1e3),  # joint_2: 1 radian to 0.001 degree
            int(states[2] * 180 / np.pi * 1e3),  # joint_3: 1 radian to 0.001 degree
            int(states[3] * 180 / np.pi * 1e3),  # joint_4: 1 radian to 0.001 degree
            int(states[4] * 180 / np.pi * 1e3),  # joint_5: 1 radian to 0.001 degree
            int(states[5] * 180 / np.pi * 1e3),  # joint_6: 1 radian to 0.001 degree
            int(states[6] / 1.6 * 60000)   # gripper: [0, 1.6] in [0, 60000]
        ]


class PiperEndEffectorStandardization(BaseStandardization):
    """
    Standardization for Piper end effector states and actions.
    """

    def input_transform(self, states):
        return [
            states[0] * 1e-6,  # x: 0.001mm to 1m
            states[1] * 1e-6,  # y: 0.001mm to 1m
            states[2] * 1e-6,  # z: 0.001mm to 1m
            states[3] * 1e-3 * np.pi / 180,  # roll: 0.001 degree to 1 radian
            states[4] * 1e-3 * np.pi / 180,  # pitch: 0.001 degree to 1 radian
            states[5] * 1e-3 * np.pi / 180,  # yaw: 0.001 degree to 1 radian
            states[6] / 60000.0 * 1.6  # gripper: [0, 60000] to [0, 1.6]
        ]

    def output_transform(self, states):
        return [
            int(states[0] * 1e6),  # x: 1m to 0.001mm
            int(states[1] * 1e6),  # y: 1m to 0.001mm
            int(states[2] * 1e6),  # z: 1m to 0.001mm
            int(states[3] * 180 / np.pi * 1e3),  # roll: 1 radian to 0.001 degree
            int(states[4] * 180 / np.pi * 1e3),  # pitch: 1 radian to 0.001 degree
            int(states[5] * 180 / np.pi * 1e3),  # yaw: 1 radian to 0.001 degree
            int(states[6] / 1.6 * 60000)   # gripper: [0, 1.6] in [0, 60000]
        ]


class BiStandardization(BaseStandardization):
    """
    Bi-standardization class that can switch between dummy and Piper standardization.
    """

    def __init__(self, standardization):
        self.standardization = get_standardization(standardization)

    def input_transform(self, states):
        left_states = states[:7]
        right_states = states[7:]
        return self.standardization.input_transform(left_states) + self.standardization.input_transform(right_states)

    def output_transform(self, states):
        left_states = states[:7]
        right_states = states[7:]
        return self.standardization.output_transform(left_states) + self.standardization.output_transform(right_states)


def get_standardization(standardization_type: str) -> BaseStandardization:
    """
    Factory function to get the standardization class based on the type.
    """

    multi_arm = False
    if standardization_type.startswith("bi_"):
        multi_arm = True
        standardization_type = standardization_type[3:]

    if standardization_type == "dummy":
        standardization = DummyStandardization()
    elif standardization_type == "piper":
        standardization = PiperJointStandardization()
    elif standardization_type == "piper_end_effector":
        standardization = PiperEndEffectorStandardization()
    elif standardization_type == "realman":
        # TODO
        standardization = DummyStandardization()
    elif standardization_type == "realman_end_effector":
        # TODO
        standardization = DummyStandardization()
    else:
        raise ValueError(f"Unknown standardization type: {standardization_type}")

    if multi_arm:
        standardization = BiStandardization(standardization)
    
    return standardization
