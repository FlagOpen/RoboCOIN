import math
from dataclasses import dataclass, field
from typing import Any, List, Dict


@dataclass
class TaskConfig:
    scene: str = "a blue towel is place on a white table, the towel is on the middle."
    task: str = "fold the towel."
    subtasks: List[str] = field(default_factory=lambda: [
        "right gripper catch the bottom right corner of the towel",
        "right grippers arrange the tilted towel in a horizontal position",
        "left gripper catch the top left corner of the towel",
        "left gripper fold the towel to the right",
        "end",
    ])
    operaters: List[Dict] = field(default_factory=lambda: [
        {
            'type': 'position',
            'name': 'position_left',
            'window_size': 1,
            'state_key': 'observation.state',
            'xyz_range': (0, 3),
        }, {
            'type': 'position',
            'name': 'position_right',
            'window_size': 1,
            'state_key': 'observation.state',
            'xyz_range': (7, 10),
        }, {
            'type': 'position_rotation',
            'name': 'position_aligned_left',
            'window_size': 1,
            'position_key': 'position_left',
            'rotation_euler': (0, 0, 0.5 * math.pi),
        }, {
            'type': 'position_rotation',
            'name': 'position_aligned_right',
            'window_size': 1,
            'position_key': 'position_right',
            'rotation_euler': (0, 0, 0.5 * math.pi),
        }, {
            'type': 'movement',
            'name': 'movement_left',
            'window_size': 3,
            'position_key': 'position_aligned_left',
        }, {
            'type': 'movement',
            'name': 'movement_right',
            'window_size': 3,
            'position_key': 'position_aligned_right',
        },{
            'type': 'movement_summary',
            'name': 'movement_summary_left',
            'movement_key': 'movement_left',
            'threshold': 2e-3,
        }, {
            'type': 'movement_summary',
            'name': 'movement_summary_right',
            'movement_key': 'movement_right',
            'threshold': 2e-3,
        }, 
    ])