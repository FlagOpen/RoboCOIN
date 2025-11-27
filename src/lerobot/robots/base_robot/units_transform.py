"""
Unit transformation classes for length and angle measurements.
"""

import math
import numpy as np
from typing import List


class LengthTransform:
    """
    Unit conversion for length measurements.
    The general unit is meter (m), 
    input values are converted from the specified unit to meters,
    and output values are converted from meters to the specified unit.
    Params:
    - from_unit: str, unit of the input value, choices are:
        - '001mm' (micrometers)
        - 'mm' (millimeters)
        - 'cm' (centimeters)
        - 'm' (meters)
    e.g.
    ```python
    lt = LengthTransform('cm')
    value_in_m = lt.input_transform(10)  # convert 10 cm to meters
    value_in_cm = lt.output_transform(0.1)  # convert 0.1 m to centimeters
    ```
    """

    def __init__(self, from_unit: str) -> None:
        assert from_unit in ['001mm', 'mm', 'cm', 'm'], 'Unit must be one of 001mm, mm, cm, m'
        self.from_unit = from_unit

    def input_transform(self, value: float) -> float:
        """Convert input value from specified unit to meters"""
        unit_map = {
            '001mm': 0.000001, # Micrometers to meters
            'mm': 0.001,       # Millimeters to meters
            'cm': 0.01,        # Centimeters to meters
            'm': 1.0,          # Meters to meters (no conversion)
        }
        return value * unit_map[self.from_unit]
    
    def output_transform(self, value: float) -> float:
        """Convert output value from meters to specified unit"""
        unit_map = {
            '001mm': 1000000.0, # Meters to micrometers
            'mm': 1000.0,       # Meters to millimeters
            'cm': 100.0,        # Meters to centimeters
            'm': 1.0,           # Meters to meters (no conversion)
        } 
        return value * unit_map[self.from_unit]


class AngleTransform:
    """
    Unit conversion for angle measurements.
    The general unit is radian,
    input values are converted from the specified unit to radians,
    and output values are converted from radians to the specified unit.
    Params:
    - from_unit: str, unit of the input value, choices are:
        - '001degree' (0.001 degrees)
        - 'degree' (degrees)
        - 'radian' (radians)
    e.g.:
    ```python
    at = AngleTransform('degree')
    value_in_rad = at.input_transform(90)  # convert 90 degrees to radians
    value_in_deg = at.output_transform(math.pi / 2)  # convert pi/2 radians to degrees
    ```
    """

    def __init__(self, from_unit: str) -> None:
        assert from_unit in ['001degree', 'degree', 'radian'], 'Unit must be one of 001degree, degree, radian'
        self.from_unit = from_unit

    def input_transform(self, value: float) -> float:
        """Convert input value from specified unit to radians"""
        unit_map = {
            '001degree': math.pi / 180000.0, # 0.001 degrees to radians
            'degree': math.pi / 180.0,       # degrees to radians
            'radian': 1.0,                   # radians to radians (no conversion)
        }
        return value * unit_map[self.from_unit]
   
    def output_transform(self, value: float) -> float:
        """Convert output value from radians to specified unit"""
        unit_map = {
            '001degree': 180000.0 / math.pi, # Radians to 0.001 degrees
            'degree': 180.0 / math.pi,       # Radians to degrees
            'radian': 1.0,                   # Radians to radians (no conversion)
        }
        return value * unit_map[self.from_unit]


class UnitsTransform:
    """
    Apply multiple unit transformations to an array of values.
    Params:
    - from_units: list of str, units for each value in the input array.
    e.g.:
    ```python
    ut = UnitsTransform(['cm', 'degree', 'm', 'radian'])
    # values_in_m_rad: [0.1, pi/2, 0.5, pi/2]
    values_in_m_rad = ut.input_transform([10, 90, 0.5, math.pi / 2])
    # values_in_original_units: [10, 90, 0.5, pi/2]
    values_in_original_units = ut.output_transform(values_in_m_rad)
    ```
    """

    def __init__(self, from_units: List[str]) -> None:
        self.transforms = []
        for unit in from_units:
            if unit in ['mm', 'cm', 'm']:
                self.transforms.append(LengthTransform(unit))
            elif unit in ['degree', 'radian']:
                self.transforms.append(AngleTransform(unit))
            else:
                raise ValueError(f"Unsupported unit: {unit}")
            
    def input_transform(self, values: np.ndarray) -> np.ndarray:
        """Convert input array from original units to base units (meters/radians)"""
        assert len(values) == len(self.transforms), "Length of values must match length of transforms"
        return np.array([self.transforms[i].input_transform(values[i]) for i in range(len(values))])
    
    def output_transform(self, values: np.ndarray) -> np.ndarray:
        """Convert output array from base units (meters/radians) back to original units"""
        assert len(values) == len(self.transforms), "Length of values must match length of transforms"
        return np.array([self.transforms[i].output_transform(values[i]) for i in range(len(values))])