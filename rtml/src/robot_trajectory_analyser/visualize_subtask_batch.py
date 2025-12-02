#!/usr/bin/env python3
import os
import json
import glob
import subprocess
import argparse
from pathlib import Path

# All fields to be visualized (you can add or remove as needed)
FIELDS = [
    "left_stat.orientation.angular_mean_deviation",
    "left_stat.orientation.angular_variance",
    "left_stat.orientation.std",
    
    "left_stat.position.min",
    "left_stat.position.max",
    "right_stat.position.min",
    "right_stat.position.max",
    
    "left_stat.linear_acc.max",
    "right_stat.linear_acc.max",
    
    "left_stat.linear_speed.mean",
    "left_stat.linear_speed.std",
    "right_stat.linear_speed.mean",
    "right_stat.linear_speed.std",
    
    "duration",
]

def get_nested_value(data, field_path):
    """Helper to safely get nested dict value like 'left_stat.orientation.std'"""
    keys = field_path.split('.')
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for all subtasks.")
    parser.add_argument("--dataset_path", required=True, help="Base dataset path (e.g., /media/.../xray_exp)")
    parser.add_argument("--output_root", default="__logs/xray_vis_all", help="Root output directory")
    parser.add_argument("--vis_script", default="__scripts/visualize_subtask_stats.py", help="Path to visualization script")
    args = parser.parse_args()

    # Find the first JSON file in dataset_path
    json_files = glob.glob(os.path.join(args.dataset_path, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {args.dataset_path}")
    
    stats_json_path = json_files[0]
    print(f"Using stats file: {stats_json_path}")

    # Load stats
    with open(stats_json_path, 'r') as f:
        stats_list = json.load(f)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for entry in stats_list:
        subtask_name = entry["timelinelabels"][0]
        # Sanitize folder name (remove problematic chars)
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in subtask_name).strip()
        subtask_dir = output_root / safe_name
        subtask_dir.mkdir(exist_ok=True)

        print(f"Processing subtask: {subtask_name}")

        for field in FIELDS:
            value = get_nested_value(entry["stats"], field)
            if value is None:
                print(f"  ⚠️ Field {field} not found in this subtask. Skipping.")
                continue

            # Build output path: e.g., __logs/xray_vis_all/Grasp_the_bread/left_stat.orientation.std.png
            field_clean = field.replace(".", "_")
            output_img = subtask_dir / f"{field_clean}.png"

            cmd = [
                "python", args.vis_script,
                "--subtask", subtask_name,
                "--path", args.dataset_path,
                "--field", field,
                "--output", str(output_img)
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"  ✅ Generated: {output_img}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to generate {field}: {e}")

    print(f"\n✅ All visualizations saved under: {output_root}")

if __name__ == "__main__":
    main()
