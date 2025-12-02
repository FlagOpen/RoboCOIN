#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml
import numpy as np


class RTMLQualityFilter:
    def __init__(self, rtml_config_path: str):
        with open(rtml_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.task_config = config['task']
        self.global_constraints = self.task_config.get('global_constraints', {})
        self.stages = self.task_config.get('stages', [])

    def _get_arm_key(self, arm: str) -> str:
        return f"{arm}_stat"

    def _check_workspace(self, stats: Dict, arm: str, ws_min: List[float], ws_max: List[float]) -> bool:
        arm_key = self._get_arm_key(arm)
        # pos_mean = stats.get(arm_key, {}).get('position', {}).get('mean')
        pos_min = stats.get(arm_key, {}).get('position', {}).get('min')
        pos_max = stats.get(arm_key, {}).get('position', {}).get('max')
        # if pos_mean is None:
        #     return False
        # pos = np.array(pos_mean)
        pos_min = np.array(pos_min)
        pos_max = np.array(pos_max)
        ws_min = np.array(ws_min)
        ws_max = np.array(ws_max)
        return np.all(pos_min >= ws_min) and np.all(pos_max <= ws_max)
    
    def _check_orientation_constraints(self, stats: Dict, arm: str, ori_cfg: Dict) -> bool:
        arm_key = self._get_arm_key(arm)
        ori_data = stats.get(arm_key, {}).get('orientation', {})
        
        # 1. Check upper bound of angular_mean_deviation
        if 'angular_mean_deviation_max' in ori_cfg:
            dev = ori_data.get('angular_mean_deviation', np.inf)
            if dev > ori_cfg['angular_mean_deviation_max']:
                return False

        # 2. Check upper bound of orientation std (three axes)
        if 'std_max' in ori_cfg:
            std = ori_data.get('std')
            if std is None:
                return False
            if not np.all(np.array(std) <= np.array(ori_cfg['std_max'])):
                return False

        # Optional: Future extensions could include mean_quat distance, angular_variance, etc.
        if 'angular_variance_max' in ori_cfg:
            var = ori_data.get('angular_variance', np.inf)
            if var > ori_cfg['angular_variance_max']:
                return False

        return True


    def _check_velocity_constraints(self, stats: Dict, arm: str, vel_cfg: Dict) -> bool:
        arm_key = self._get_arm_key(arm)
        linear_speed = stats.get(arm_key, {}).get('linear_speed', {})
        angular_speed = stats.get(arm_key, {}).get('angular_speed', {})

        # Check maximum linear velocity
        if 'max' in vel_cfg.get('linear', {}):
            if linear_speed.get('max', np.inf) > vel_cfg['linear']['max']:
                return False
        if 'mean_max' in vel_cfg.get('linear', {}):
            if linear_speed.get('mean', np.inf) > vel_cfg['linear']['mean_max']:
                return False
        if 'mean_min' in vel_cfg.get('linear', {}):
            if linear_speed.get('mean', -np.inf) < vel_cfg['linear']['mean_min']:
                return False
        if 'std_max' in vel_cfg.get('linear', {}):
            if linear_speed.get('std', np.inf) > vel_cfg['linear']['std_max']:
                return False

        # Angular velocity
        if 'max' in vel_cfg.get('angular', {}):
            if angular_speed.get('max', np.inf) > vel_cfg['angular']['max']:
                return False
        if 'mean_max' in vel_cfg.get('angular', {}):
            if angular_speed.get('mean', np.inf) > vel_cfg['angular']['mean_max']:
                return False

        return True

    def _check_acceleration_constraints(self, stats: Dict, arm: str, acc_cfg: Dict) -> bool:
        arm_key = self._get_arm_key(arm)
        linear_acc = stats.get(arm_key, {}).get('linear_acc', {})
        angular_acc = stats.get(arm_key, {}).get('angular_acc', {})

        if 'max' in acc_cfg.get('linear', {}):
            if linear_acc.get('max', np.inf) > acc_cfg['linear']['max']:
                return False
        if 'max' in acc_cfg.get('angular', {}):
            if angular_acc.get('max', np.inf) > acc_cfg['angular']['max']:
                return False
        return True

    def _check_position_std(self, stats: Dict, arm: str, std_max: List[float]) -> bool:
        arm_key = self._get_arm_key(arm)
        pos_std = stats.get(arm_key, {}).get('position', {}).get('std')
        if pos_std is None:
            return False
        return np.all(np.array(pos_std) <= np.array(std_max))

    def _check_temporal(self, stats: Dict, temp_cfg: Dict) -> bool:
        duration = stats.get('duration', 0)
        if 'duration_min' in temp_cfg and duration < temp_cfg['duration_min']:
            return False
        if 'duration_max' in temp_cfg and duration > temp_cfg['duration_max']:
            return False
        return True

    def _check_idle_arm(self, stats: Dict, idle_cfg: Dict) -> bool:
        arm = idle_cfg['arm']  # "left" or "right"
        arm_key = self._get_arm_key(arm)
        speed_mean = stats.get(arm_key, {}).get('linear_speed', {}).get('mean', np.inf)
        threshold = idle_cfg.get('velocity_linear_mean_max', 0.01)
        return speed_mean <= threshold

    def _apply_constraints(self, stats: Dict, constraints: Dict, arms: List[str] = ["left", "right"]) -> bool:
        # 1. Workspace
        ws = constraints.get('workspace', {})
        for arm in arms:
            if arm in ws:
                if not self._check_workspace(stats, arm, ws[arm]['min'], ws[arm]['max']):
                    return False

        # 2. Velocity
        if 'velocity' in constraints:
            for arm in arms:
                if not self._check_velocity_constraints(stats, arm, constraints['velocity']):
                    return False

        # 3. Acceleration
        if 'acceleration' in constraints:
            for arm in arms:
                if not self._check_acceleration_constraints(stats, arm, constraints['acceleration']):
                    return False

        # 4. Position std (for stability)
        if 'position_std' in constraints:
            ps = constraints['position_std']
            for arm in arms:
                if arm in ps:
                    if not self._check_position_std(stats, arm, ps[arm]['max']):
                        return False

        # 5. Temporal
        if 'temporal' in constraints:
            if not self._check_temporal(stats, constraints['temporal']):
                return False

        # 6. Idle arm
        if 'idle_arm' in constraints:
            if not self._check_idle_arm(stats, constraints['idle_arm']):
                return False
            
        # 7. Orientation constraints (new)
        if 'orientation' in constraints:
            ori_cfg = constraints['orientation']
            for arm in arms:
                if arm in ori_cfg:
                    if not self._check_orientation_constraints(stats, arm, ori_cfg[arm]):
                        return False

        return True

    def is_high_quality(self, step: Dict[str, Any]) -> bool:
        timelinelabels = step.get("timelinelabels", [])
        subtask_name = timelinelabels[0] if timelinelabels else ""
        # subtask_name = step.get("subtask", "")
        stats = step.get("stats", {})
        if not stats:
            return True

        # Find matching stage
        matched_stage = None
        for stage in self.stages:
            if stage.get("match_subtask") == subtask_name:
                matched_stage = stage
                break

        # Build final constraints: global + stage-specific
        final_constraints = {}
        if self.global_constraints:
            final_constraints.update(self.global_constraints)

        if matched_stage:
            stage_constraints = matched_stage.get("constraints")
            if stage_constraints is not None:
                # Merge even if it's an empty dict (though it has no effect)
                for key, val in stage_constraints.items():
                    final_constraints[key] = val

        # Default to checking both arms
        arms_to_check = ["left", "right"]

        # Apply constraint checks
        return self._apply_constraints(stats, final_constraints, arms_to_check)


def filter_with_rtml(
    data_dir: str,
    rtml_path: str,
) -> tuple:
    filter_obj = RTMLQualityFilter(rtml_path)
    data_path = Path(data_dir)
    good_episodes = []
    good_subtask_indices = {}

    ep_files = sorted(data_path.glob("*episode_*.json"))
    print(f"🔍 Found {len(ep_files)} raw collected episodes")
    for ep_file in ep_files:
        try:
            with open(ep_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Skipping corrupted file {ep_file}: {e}")
            continue

        indices = []
        good_indices = []
        for idx, step in enumerate(data):
            # indices.append(step['subtask'])
            indices.append(step['timelinelabels'][0])
            if filter_obj.is_high_quality(step):
                # good_indices.append(idx)
                # good_indices.append(step['subtask'])
                good_indices.append(step['timelinelabels'][0])
        # print(len(good_indices), len(data))
        # print(good_indices, indices)
        if len(good_indices) == len(data): # Only consider episode valid if all steps pass
            good_episodes.append(ep_file.name)
            good_subtask_indices[ep_file.name] = good_indices


    return ep_files, good_episodes, good_subtask_indices


def main():
    parser = argparse.ArgumentParser(description="Filter high-quality robot data using RTML configuration file")
    parser.add_argument("--input", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--rtml", type=str, required=True, help="RTML configuration file path (YAML)")
    parser.add_argument("--output_file", type=str, default=None, help="Output filename")

    args = parser.parse_args()

    ep_files, good_eps, good_idx = filter_with_rtml(
        data_dir=args.input,
        rtml_path=args.rtml,
    )

    print(f"\n✅ Found {len(good_eps)} high-quality episodes")
    print(f"Number removed:", len(ep_files) - len(good_eps))

    # Write good_eps to output_file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for ep in good_eps:
                f.write(ep + '\n')
    print(f"Output file: {args.output_file}")
if __name__ == "__main__":
    main()
