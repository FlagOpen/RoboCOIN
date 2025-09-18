import math
from dataclasses import dataclass, field
from typing import Any, List, Dict


@dataclass
class TaskConfig:
    scene: str = "a black box and a pink peach are place on a white table, the box is on the left and the peach is on the right."
    task: str = "open the box and put the peach in it."
    subtasks: List[str] = field(default_factory=lambda: [
        "left gripper catch box",
        "left gripper open box",
        "right gripper catch peach",
        "right gripper move peach in box and release",
        "left gripper release to close box",
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