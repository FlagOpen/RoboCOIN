import math
import numpy as np


class LengthTransform:
    def __init__(self, from_unit: str):
        assert from_unit in ['mm', 'cm', 'm'], 'Unit must be one of mm, cm, m'
        self.from_unit = from_unit

    def input_transform(self, value: float) -> float:
        # convert to m
        unit_map = {
            'mm': 0.001,
            'cm': 0.01,
            'm': 1.0,
        }
        return value * unit_map[self.from_unit]
    
    def output_transform(self, value: float) -> float:
        # convert from m
        unit_map = {
            'mm': 1000.0,
            'cm': 100.0,
            'm': 1.0,
        }
        return value * unit_map[self.from_unit]


class AngleTransform:
    def __init__(self, from_unit: str):
        assert from_unit in ['degree', 'radian'], 'Unit must be one of degree, radian'
        self.from_unit = from_unit

    def input_transform(self, value: float) -> float:
        # convert to radian
        unit_map = {
            'degree': math.pi / 180.0,
            'radian': 1.0,
        }
        return value * unit_map[self.from_unit]
   
    def output_transform(self, value: float) -> float:
        # convert from radian
        unit_map = {
            'degree': 180.0 / math.pi,
            'radian': 1.0,
        }
        return value * unit_map[self.from_unit]


class UnitsTransform:
    def __init__(self, from_units: list[str]):
        self.transforms = []
        for unit in from_units:
            if unit in ['mm', 'cm', 'm']:
                self.transforms.append(LengthTransform(unit))
            elif unit in ['degree', 'radian']:
                self.transforms.append(AngleTransform(unit))
            else:
                raise ValueError(f"Unsupported unit: {unit}")
            
    def input_transform(self, values: np.ndarray) -> np.ndarray:
        assert len(values) == len(self.transforms), "Length of values must match length of transforms"
        return np.array([self.transforms[i].input_transform(values[i]) for i in range(len(values))])
    
    def output_transform(self, values: np.ndarray) -> np.ndarray:
        assert len(values) == len(self.transforms), "Length of values must match length of transforms"
        return np.array([self.transforms[i].output_transform(values[i]) for i in range(len(values))])