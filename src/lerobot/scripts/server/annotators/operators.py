import copy
import json
import numpy as np
import os
import yaml
from abc import ABC, abstractmethod

from .transforms import (
    quaternion_to_euler,
    matrix6d_to_euler,
    matrix_to_euler,
    position_subtract,
    position_rotate,
    euler_add,
)


def get_default_lerobot_root():
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'lerobot')


class BaseOperator(ABC):
    def __init__(
        self, 
        name,
        window_size=1, 
    ):
        self.name = name
        self.window_size = window_size
    
    @abstractmethod
    def _operate(self, frame_window, annotation_window):
        pass

    def operate(self, episode, annotations):
        frame_windows, annotation_windows = self._split_windows(episode, annotations)

        results = []
        for frame_window, annotation_window in zip(frame_windows, annotation_windows):
            result = self._operate(frame_window, annotation_window)
            results.append(result)

        for result, annotation in zip(results, annotations):
            annotation[self.name] = result

        return annotations

    def _split_windows(self, episode, annotations):
        # e.g. [a, b, c, d, e], window_size=3 
        # -> [[a,a,a], [a,a,b], [a,b,c], [b,c,d], [c,d,e]]
        if self.window_size <= 1:
            return [[frame] for frame in episode], [[annotation] for annotation in annotations]
        
        frame_windows, annotation_windows = [], []
        for i in range(len(episode)):
            start = max(0, i - self.window_size + 1)
            frame_window = episode[start:i + 1]
            annotation_window = annotations[start:i + 1]

            if len(frame_window) < self.window_size:
                frame_window = [copy.deepcopy(frame_window[0])] * (self.window_size - len(frame_window)) + frame_window
                annotation_window = [copy.deepcopy(annotation_window[0])] * (self.window_size - len(annotation_window)) + annotation_window

            frame_windows.append(frame_window)
            annotation_windows.append(annotation_window)

        return frame_windows, annotation_windows


class PositionOperator(BaseOperator):
    def __init__(
        self,
        state_key,
        xyz_range,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state_key = state_key
        self.xyz_range = xyz_range
    
    def _operate(self, frame_window, annotation_window):
        curr_xyz = frame_window[-1][self.state_key][self.xyz_range[0]:self.xyz_range[1]]
        return list(curr_xyz)


class AngleOperator(BaseOperator):
    def __init__(
        self,
        state_key,
        rpy_range,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state_key = state_key
        self.rpy_range = rpy_range
    
    def _operate(self, frame_window, annotation_window):
        curr_rpy = frame_window[-1][self.state_key][self.rpy_range[0]:self.rpy_range[1]]

        if len(curr_rpy) == 4:
            curr_rpy = quaternion_to_euler(curr_rpy)
        if len(curr_rpy) == 6:
            curr_rpy = matrix6d_to_euler(curr_rpy)
        elif len(curr_rpy) == 9:
            curr_rpy = matrix_to_euler(curr_rpy)
        
        return list(curr_rpy)


class GripperOperator(BaseOperator):
    def __init__(
        self,
        state_key,
        gripper_indice,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state_key = state_key
        self.gripper_indice = gripper_indice
    
    def _operate(self, frame_window, annotation_window):
        curr_gripper = frame_window[-1][self.state_key][self.gripper_indice]
        return curr_gripper


class PositionRotationOperator(BaseOperator):
    def __init__(
        self,
        position_key,
        rotation_euler=(0, 0, 0),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.position_key = position_key
        self.rotation_euler = rotation_euler
    
    def _operate(self, frame_window, annotation_window):
        curr_pos = annotation_window[-1][self.position_key]
        aligned_pos = position_rotate(curr_pos, self.rotation_euler)
        return aligned_pos


class AngleRotationOperator(BaseOperator):
    def __init__(
        self,
        angle_key,
        rotation_euler=(0, 0, 0),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.angle_key = angle_key
        self.rotation_euler = rotation_euler
    
    def _operate(self, frame_window, annotation_window):
        curr_rpy = annotation_window[-1][self.angle_key]
        aligned_rpy = euler_add(curr_rpy, self.rotation_euler)
        return aligned_rpy


class MovementOperator(BaseOperator):
    def __init__(
        self,
        position_key,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.position_key = position_key
    
    def _operate(self, frame_window, annotation_window):
        if self.window_size < 2:
            return 0.0
        
        prev_xyz = annotation_window[0][self.position_key]
        curr_xyz = annotation_window[-1][self.position_key]
        return position_subtract(curr_xyz, prev_xyz)


class GripperMovementOperator(BaseOperator):
    def __init__(
        self,
        gripper_key,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gripper_key = gripper_key
    
    def _operate(self, frame_window, annotation_window):
        if self.window_size < 2:
            return 0.0
        
        prev_gripper = annotation_window[0][self.gripper_key]
        curr_gripper = annotation_window[-1][self.gripper_key]
        return curr_gripper - prev_gripper


class VelocityOperator(BaseOperator):
    def __init__(
        self,
        movement_key,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.movement_key = movement_key

    def _operate(self, frame_window, annotation_window):
        return np.linalg.norm(annotation_window[-1][self.movement_key]).item()


class AccelerationOperator(BaseOperator):
    def __init__(
        self,
        vel_key,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vel_key = vel_key

    def _operate(self, frame_window, annotation_window):
        if self.window_size < 2:
            return 0.0
        
        prev_vel = annotation_window[0][self.vel_key]
        curr_vel = annotation_window[-1][self.vel_key]
        accel = (curr_vel - prev_vel)
        return accel


class GripperSummaryOperator(BaseOperator):
    def __init__(
        self,
        gripper_key,
        threshold=0.01,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gripper_key = gripper_key
        self.threshold = threshold

    def _operate(self, frame_window, annotation_window):
        gripper = annotation_window[-1][self.gripper_key]
        if gripper > self.threshold:
            return 'open'
        else:
            return 'closed'


class MovementSummaryOperator(BaseOperator):
    def __init__(
        self,
        movement_key,
        threshold=1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.movement_key = movement_key
        self.threshold = threshold

    def _operate(self, frame_window, annotation_window):
        dx, dy, dz = annotation_window[-1][self.movement_key]
        if max(abs(dx), abs(dy), abs(dz)) < self.threshold:
            return 'stationary'
        
        if abs(dx) > abs(dy) and abs(dx) > abs(dz):
            if dx > 0:
                return 'right'
            else:
                return 'left'
        elif abs(dy) > abs(dx) and abs(dy) > abs(dz):
            if dy > 0:
                return 'forward'
            else:
                return 'backward'
        elif abs(dz) > abs(dx) and abs(dz) > abs(dy):
            if dz > 0:
                return 'up'
            else:
                return 'down'
        else:
            return 'stationary'


class GripperMovementSummaryOperator(BaseOperator):
    def __init__(
        self,
        gripper_movement_key,
        threshold=-1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gripper_movement_key = gripper_movement_key
        self.threshold = threshold

    def _operate(self, frame_window, annotation_window):
        movement = annotation_window[-1][self.gripper_movement_key]
        if movement > self.threshold:
            return 'opening'
        elif movement < - self.threshold:
            return 'closing'
        else:
            return 'holding'


class VelocitySummaryOperator(BaseOperator):
    def __init__(
        self,
        velocity_key,
        slow_threshold=1e-3,
        fast_threshold=5e-3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vel_key = velocity_key
        self.slow_threshold = slow_threshold
        self.fast_threshold = fast_threshold

    def _operate(self, frame_window, annotation_window):
        vel = annotation_window[-1][self.vel_key]
        if vel < self.slow_threshold:
            return 'stationary'
        elif vel < self.fast_threshold:
            return 'slow'
        else:
            return 'fast'


class AccelerationSummaryOperator(BaseOperator):
    def __init__(
        self,
        acceleration_key,
        threshold=1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.accel_key = acceleration_key
        self.threshold = threshold

    def _operate(self, frame_window, annotation_window):
        accel = annotation_window[-1][self.accel_key]
        if accel < -self.threshold:
            return 'decelerating'
        elif accel > self.threshold:
            return 'accelerating'
        else:
            return 'constant'


class KeepAnnotationOperator(BaseOperator):
    def __init__(
        self,
        keys,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.keys = keys
    
    def _operate(self, frame_window, annotation_window):
        curr_annotation = annotation_window[-1]
        return {key: curr_annotation[key] for key in self.keys if key in curr_annotation}


class SceneDescriptionOperator(BaseOperator):
    def __init__(
        self, 
        scene_description_dir,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._scene_descriptions = self._load_scene_descriptions(scene_description_dir)
        self._task_meta = self._load_task_meta(os.path.join(get_default_lerobot_root(), self.repo_id, 'meta', 'tasks.jsonl'))
    
    def _operate(self, frame_window, annotation_window):
        task_index = annotation_window[-1]['task_index']
        task = self._task_meta[task_index].replace('.', '')
        return self._scene_descriptions[task]

    def _load_scene_descriptions(self, dir):
        desc = {}
        for file in os.listdir(dir):
            if file.endswith('.yaml') or file.endswith('.yml'):
                path = os.path.join(dir, file)
                with open(path) as f:
                    data = yaml.safe_load(f)
                desc[file.split('.')[0]] = data['scene']
        return desc
    
    def _load_task_meta(self, path):
        with open(path) as f:
            lines = f.readlines()
        task_map = {}
        for line in lines:
            data = json.loads(line)
            task_map[data['task_index']] = data['task']
        return task_map


class SubtaskOperator(BaseOperator):
    def __init__(
        self,
        subtask_annotation_path,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._annotations = self._load_subtask_annotations(subtask_annotation_path)
    
    def _operate(self, frame_window, annotation_window):
        episode_index = annotation_window[-1]['episode_index']
        frame_index = annotation_window[-1]['frame_index']
        subtasks = self._annotations[episode_index]
        for start, end, label in subtasks:
            if start <= frame_index <= end:
                return label
        return 'none'
    
    def _load_subtask_annotations(self, path):
        with open(path) as f:
            data = json.load(f)
        annotations = {}
        for episode in data:
            episode_index = int(episode['video'].split('episode_')[1].split('.')[0])
            subtasks = episode['videoLabels']
            subtasks = [
                (
                    subtask['ranges'][0]['start'] - 1,
                    subtask['ranges'][0]['end'] - 1,
                    subtask['timelinelabels'][0].lower()
                )
                for subtask in subtasks
            ]
            annotations[episode_index] = subtasks
        return annotations


class Pipeline:
    def __init__(self, operators):
        self.operators = operators
    
    def __call__(self, episode):
        annotations = [{"frame_index": i} for i in range(len(episode))]
        for operator in self.operators:
            annotations = operator.operate(episode, annotations)
        return annotations


def make_operator_from_config(config):
    op_type = config['type']
    del config['type']

    if op_type == 'position':
        return PositionOperator(**config)
    elif op_type == 'angle':
        return AngleOperator(**config)
    elif op_type == 'gripper':
        return GripperOperator(**config)
    elif op_type == 'position_rotation':
        return PositionRotationOperator(**config)
    elif op_type == 'angle_rotation':
        return AngleRotationOperator(**config)
    elif op_type == 'movement':
        return MovementOperator(**config)
    elif op_type == 'gripper_movement':
        return GripperMovementOperator(**config)
    elif op_type == 'velocity':
        return VelocityOperator(**config)
    elif op_type == 'acceleration':
        return AccelerationOperator(**config)
    elif op_type == 'gripper_summary':
        return GripperSummaryOperator(**config)
    elif op_type == 'movement_summary':
        return MovementSummaryOperator(**config)
    elif op_type == 'gripper_movement_summary':
        return GripperMovementSummaryOperator(**config)
    elif op_type == 'velocity_summary':
        return VelocitySummaryOperator(**config)
    elif op_type == 'acceleration_summary':
        return AccelerationSummaryOperator(**config)
    elif op_type == 'keep_annotation':
        return KeepAnnotationOperator(**config)
    elif op_type == 'scene_description':
        return SceneDescriptionOperator(**config)
    elif op_type == 'subtask':
        return SubtaskOperator(**config)
    else:
        raise ValueError(f"Unknown operator type: {op_type}")


def make_operators_pipeline(configs):
    operators = []
    for config in configs:
        operator = make_operator_from_config(config)
        operators.append(operator)
    return Pipeline(operators)