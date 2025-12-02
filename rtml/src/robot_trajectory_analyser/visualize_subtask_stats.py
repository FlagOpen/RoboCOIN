#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_field_path(field: str) -> List[str]:
    """Convert a field path like 'left_stat.position.mean' into ['left_stat', 'position', 'mean']"""
    return field.strip().split('.')


def get_nested_value(data: dict, keys: List[str]) -> Any:
    """Safely retrieve a value from a nested dictionary"""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data


def collect_values_from_episodes(
    data_dir: str, subtask_name: str, field_path: List[str]
) -> List[Union[float, List[float]]]:
    """Collect values of a specified field for a given subtask from all episode files"""
    values = []
    episode_files = os.listdir(data_dir)
    for ep_file in episode_files:
        ep_file = os.path.join(data_dir, ep_file)
        try:
            with open(ep_file, 'r') as f:
                episode_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read {ep_file}: {e}")
            continue

        for step in episode_data:
            if not isinstance(step, dict):
                continue
            if step.get("timelinelabels")[0] != subtask_name:
                continue

            stats = step.get("stats", {})
            if not stats:
                continue

            val = get_nested_value(stats, field_path)
            if val is not None:
                values.append(val)

    return values


def flatten_values(values: List[Union[float, List[float]]]) -> np.ndarray:
    """Convert a list of values into a numpy array. If values are vectors, stack them column-wise."""
    if not values:
        return np.array([])

    # Check whether the first valid value is a scalar or vector
    first = None
    for v in values:
        if v is not None:
            first = v
            break
    if first is None:
        return np.array([])

    if isinstance(first, (int, float)):
        return np.array([v for v in values if v is not None])
    elif isinstance(first, (list, tuple)) and len(first) > 0:
        # Assume all vectors have the same length
        arr = np.array([v for v in values if v is not None])
        if arr.ndim == 2:
            return arr  # shape (N, D)
        else:
            return arr.reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported data type: {type(first)}")


def plot_distribution(data: np.ndarray, field: str, subtask: str, output_path: str):
    """Plot the distribution of the given data"""
    if data.size == 0:
        print("❌ No data found!")
        return

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
        # Scalar or single-column data
        flat_data = data.flatten()
        sns.histplot(flat_data, kde=True, stat="density", bins=30)
        plt.title(f"Distribution of '{field}' in subtask '{subtask}'")
        plt.xlabel(field)
        plt.ylabel("Density")
    else:
        # Multi-dimensional vector (e.g., x, y, z components of position.mean)
        dims = data.shape[1]
        labels = [f"{field}[{i}]" for i in range(dims)]
        for i in range(dims):
            sns.kdeplot(data[:, i], label=labels[i], fill=True)
        plt.title(f"Distribution of '{field}' (per dimension) in subtask '{subtask}'")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"✅ Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the distribution of a statistical field for a specified subtask")
    parser.add_argument("--subtask", type=str, required=True, help="Subtask name, e.g., 'Grasp the pink doll with the left gripper'")
    parser.add_argument("--path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--field", type=str, required=True, help="Field path to visualize, e.g., 'left_stat.position.mean'")
    parser.add_argument("--output", type=str, default="distribution.png", help="Output image file path")

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise ValueError(f"Directory does not exist: {args.path}")

    field_keys = parse_field_path(args.field)
    print(f"🔍 Collecting field '{args.field}' for subtask='{args.subtask}' ...")
    raw_values = collect_values_from_episodes(args.path, args.subtask, field_keys)

    if not raw_values:
        print("❌ No matching data found! Please check the subtask name or field path.")
        return

    data_array = flatten_values(raw_values)
    plot_distribution(data_array, args.field, args.subtask, args.output)


if __name__ == "__main__":
    main()
