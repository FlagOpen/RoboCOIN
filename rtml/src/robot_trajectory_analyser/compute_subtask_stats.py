#!/usr/bin/env python3
import argparse
import os
import json
import lerobot
import numpy as np
import pinocchio as pin
from pathlib import Path
from pprint import pprint
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Robot parameters: used for forward kinematics to compute end-effector workspace

# ================================
# Configuration Parameters
# ================================
URDF_PATH = "../../assets/g1/g1_body29_hand14.urdf"
EE_LEFT_LINK = "left_hand_palm_joint"
EE_RIGHT_LINK = "right_hand_palm_joint"
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

# Mapping joint names to indices (for extracting relevant joints from arbitrary qpos)
name_to_index = {name: idx for idx, name in enumerate(ARM_JOINT_NAMES)}

# Load robot model
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()

# Get end-effector frame IDs
try:
    ee_left_id = model.getFrameId(EE_LEFT_LINK)
    ee_right_id = model.getFrameId(EE_RIGHT_LINK)
except Exception as e:
    raise ValueError(f"Link not found in URDF: {e}")

# Utility function: compute statistics for a sequence of quaternion poses

def get_stats_quat(quaternions):
    """
    Compute statistical metrics for a sequence of quaternions.

    Args:
        quaternions: numpy array of shape (N, 4), format [w, x, y, z]

    Returns:
        Dictionary containing quaternion-based statistics.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    quaternions = np.array(quaternions)
    N = quaternions.shape[0]
    
    if N == 0:
        return {
            "mean_quat": [1.0, 0.0, 0.0, 0.0],
            "angular_variance": 0.0,
            "angular_mean_deviation": 0.0,
            "min_roll": 0.0, "max_roll": 0.0,
            "min_pitch": 0.0, "max_pitch": 0.0,
            "min_yaw": 0.0, "max_yaw": 0.0,
            "roll_mean": 0.0, "roll_std": 0.0,
            "pitch_mean": 0.0, "pitch_std": 0.0,
            "yaw_mean": 0.0, "yaw_std": 0.0
        }
    
    # Ensure unit quaternions
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions_normalized = quaternions / norms
    
    # === 1. Compute mean quaternion ===
    def mean_quaternion(quats):
        """Compute mean quaternion using eigenvector method."""
        # Resolve sign ambiguity between q and -q
        quats = quats.copy()
        q0 = quats[0]
        for i in range(1, len(quats)):
            if np.dot(quats[i], q0) < 0:
                quats[i] = -quats[i]
        
        # Build covariance matrix
        Q = quats.T @ quats
        # Eigenvector corresponding to largest eigenvalue
        eigenvals, eigenvecs = np.linalg.eigh(Q)
        mean_q = eigenvecs[:, np.argmax(eigenvals)]
        return mean_q / np.linalg.norm(mean_q)
    
    mean_quat = mean_quaternion(quaternions_normalized)
    
    # === 2. Compute angular variance and mean deviation ===
    def quaternion_angular_distance(q1, q2):
        """Compute minimal rotation angle (in radians) between two quaternions."""
        dot_product = np.abs(np.dot(q1, q2))  # Handle q vs -q
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return 2 * np.arccos(dot_product)
    
    angles = []
    for q in quaternions_normalized:
        angle = quaternion_angular_distance(q, mean_quat)
        angles.append(angle)
    angles = np.array(angles)
    
    angular_variance = np.var(angles)
    angular_mean_deviation = np.mean(angles)
    
    # === 3. Convert to Euler angles for range statistics ===
    # scipy expects [x, y, z, w], while our data is [w, x, y, z]
    scipy_quats = np.column_stack([quaternions_normalized[:, 1],  # x
                                   quaternions_normalized[:, 2],  # y  
                                   quaternions_normalized[:, 3],  # z
                                   quaternions_normalized[:, 0]]) # w
    
    try:
        rotations = R.from_quat(scipy_quats)
        euler_angles = rotations.as_euler('xyz', degrees=False)  # roll, pitch, yaw
        
        roll = euler_angles[:, 0]
        pitch = euler_angles[:, 1]
        yaw = euler_angles[:, 2]
        
        # Handle angle wrapping (optional, simple unwrapping used here)
        def unwrap_angles(angles):
            """Simple angle unwrapping."""
            return np.unwrap(angles)
        
        roll_unwrapped = unwrap_angles(roll)
        pitch_unwrapped = unwrap_angles(pitch)
        yaw_unwrapped = unwrap_angles(yaw)
        
        # Compile statistics
        stats = {
            "mean_quat": mean_quat.tolist(),
            "angular_variance": float(angular_variance),
            "angular_mean_deviation": float(angular_mean_deviation),
            
            "min" : [float(np.min(roll)), float(np.min(pitch)), float(np.min(yaw))], # roll, pitch, yaw
            "max" : [float(np.max(roll)), float(np.max(pitch)), float(np.max(yaw))], # roll, pitch, yaw
            "mean": [float(np.mean(roll_unwrapped)), float(np.mean(pitch_unwrapped)), float(np.mean(yaw_unwrapped))], # roll, pitch, yaw
            "std": [float(np.std(roll_unwrapped)), float(np.std(pitch_unwrapped)), float(np.std(yaw_unwrapped))] # roll, pitch, yaw
        }
        
    except Exception as e:
        # Fallback if Euler conversion fails
        print(f"Warning: Euler angle conversion failed: {e}")
        stats = {
            "mean_quat": mean_quat.tolist(),
            "angular_variance": float(angular_variance),
            "angular_mean_deviation": float(angular_mean_deviation),
            "min" : [0.0, 0.0, 0.0], # roll, pitch, yaw
            "max" : [0.0, 0.0, 0.0], # roll, pitch, yaw
            "mean": [0.0, 0.0, 0.0], # roll, pitch, yaw
            "std": [0.0, 0.0, 0.0] # roll, pitch, yaw
        }
    
    return stats


# Utility function: compute statistics for end-effector trajectories, orientations, workspace, velocities, and accelerations

def compute_hand_workspace_with_velocity_acceleration_from_qpos(
    dataset, 
    model,
    ranges,
    dt=1/30,
    return_traj=False,
):
    """
    Inputs:
        - dataset: iterable where each item has ['observation']['state'] representing qpos (14-DoF arm only)
        - model: Pinocchio robot model
        - ee_left_name / ee_right_name: end-effector link names
        - dt: time step in seconds (default 0.02 for 50Hz)

    Outputs:
        Dictionary containing trajectories and statistics for left/right hand positions, orientations, velocities, and accelerations.
    """

    nq = model.nq
    nv = model.nv
    # print(f"Model nq = {nq}, nv = {nv}")

    # Initialize full state vectors
    q_full = np.zeros(nq)
    qv_full = np.zeros(nv)

    data = model.createData()

    # Store data across all timesteps (note: velocity/acceleration have fewer frames than position)
    left_positions = []
    left_quaternions = []
    left_velocities = []  # 6D spatial velocity at each valid timestep
    left_accelerations = []

    right_positions = []
    right_quaternions = []
    right_velocities = []
    right_accelerations = []
    
    left_linear_speeds = []
    left_angular_speeds = []
    left_linear_accs = []
    left_angular_accs = []
    
    right_linear_speeds = []
    right_angular_speeds = []
    right_linear_accs = []
    right_angular_accs = []
    

    N = ranges[1] - ranges[0]

    # Cache qpos sequence for finite differencing
    qpos_seq = []
    for t in tqdm(range(ranges[0], ranges[1]), desc="Cache qpos"):
        qpos_seq.append(np.array(dataset[t]['observation.state']))
        # Alternative: use a custom get_item_woimg() to skip image loading for speed

    # Estimate joint velocities via forward differencing (backward at last frame for alignment)
    qvel_estimated = []
    for t in tqdm(range(N), desc="Processing velocity&acceleration"):
        if t == N - 1:
            # Last frame: backward difference
            dq = qpos_seq[t] - qpos_seq[t-1]
        else:
            dq = qpos_seq[t+1] - qpos_seq[t]
        qvel_t = dq / dt
        qvel_estimated.append(qvel_t)
    
    prev_left_vel_sp = None
    prev_right_vel_sp = None

    for t in tqdm(range(N), desc="Processing trajectory"):
        qpos = qpos_seq[t]
        qvel = qvel_estimated[t]  # Estimated joint velocities
        # Set full state
        q_full[16:23] = qpos[:7]      # Left arm
        q_full[30:37] = qpos[7:14]   # Right arm
        qv_full[16:23] = qvel[:7]
        qv_full[30:37] = qvel[7:14]
        # Other joints remain zero or unchanged...

        # Forward kinematics (position & orientation)
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)

        # --- Position & Orientation ---
        T_left = data.oMf[ee_left_id]
        pos_left = T_left.translation.copy()
        quat_left = pin.Quaternion(T_left.rotation)
        quat_left.normalize()
        wxyz_left = [quat_left.w, quat_left.x, quat_left.y, quat_left.z]

        T_right = data.oMf[ee_right_id]
        pos_right = T_right.translation.copy()
        quat_right = pin.Quaternion(T_right.rotation)
        quat_right.normalize()
        wxyz_right = [quat_right.w, quat_right.x, quat_right.y, quat_right.z]

        # --- Velocity: compute spatial velocity using estimated qvel ---
        pin.computeForwardKinematicsDerivatives(model, data, q_full, qv_full, np.zeros(nv))
        v_left_sp = pin.getFrameVelocity(model, data, ee_left_id).vector  # 6D twist
        v_right_sp = pin.getFrameVelocity(model, data, ee_right_id).vector

        # --- Acceleration: numerical differentiation ---
        acc_left_sp = np.zeros(6)
        acc_right_sp = np.zeros(6)

        if prev_left_vel_sp is not None:
            acc_left_sp = (v_left_sp - prev_left_vel_sp) / dt
            acc_right_sp = (v_right_sp - prev_right_vel_sp) / dt
        # First frame acceleration is set to zero
        else:
            # Optional: skip first frame; here we keep it as zero
            pass
        
        # --- Compute scalar magnitudes of linear and angular speeds/accelerations ---
        # Left hand
        linear_vel_left = v_left_sp[:3]
        angular_vel_left = v_left_sp[3:]
        left_linear_speeds.append(np.linalg.norm(linear_vel_left))
        left_angular_speeds.append(np.linalg.norm(angular_vel_left))

        linear_acc_left = acc_left_sp[:3]
        angular_acc_left = acc_left_sp[3:]
        left_linear_accs.append(np.linalg.norm(linear_acc_left))
        left_angular_accs.append(np.linalg.norm(angular_acc_left))

        # Right hand
        linear_vel_right = v_right_sp[:3]
        angular_vel_right = v_right_sp[3:]
        right_linear_speeds.append(np.linalg.norm(linear_vel_right))
        right_angular_speeds.append(np.linalg.norm(angular_vel_right))

        linear_acc_right = acc_right_sp[:3]
        angular_acc_right = acc_right_sp[3:]
        right_linear_accs.append(np.linalg.norm(linear_acc_right))
        right_angular_accs.append(np.linalg.norm(angular_acc_right))

        # Store data
        left_positions.append(pos_left)
        left_quaternions.append(wxyz_left)
        left_velocities.append(v_left_sp.copy())
        left_accelerations.append(acc_left_sp)

        right_positions.append(pos_right)
        right_quaternions.append(wxyz_right)
        right_velocities.append(v_right_sp.copy())
        right_accelerations.append(acc_right_sp)

        # Update previous velocities
        prev_left_vel_sp = v_left_sp.copy()
        prev_right_vel_sp = v_right_sp.copy()

    # Convert to NumPy arrays
    left_positions = np.array(left_positions)
    left_quaternions = np.array(left_quaternions)
    left_velocities = np.array(left_velocities)
    left_accelerations = np.array(left_accelerations)

    right_positions = np.array(right_positions)
    right_quaternions = np.array(right_quaternions)
    right_velocities = np.array(right_velocities)
    right_accelerations = np.array(right_accelerations)

    # Helper function for basic statistics
    def get_stats(arr, axis=0):
        return {
            "min": np.min(arr, axis=axis).tolist(),
            "max": np.max(arr, axis=axis).tolist(),
            "mean": np.mean(arr, axis=axis).tolist(),
            "std": np.std(arr, axis=axis).tolist()
        }
        
    results =  {
        # Statistics
        "left_stat": {
            "position": get_stats(left_positions),
            "orientation": get_stats_quat(left_quaternions),  # Use quaternion-aware stats
            "linear_speed": get_stats(left_linear_speeds),
            "angular_speed": get_stats(left_angular_speeds),
            "linear_acc": get_stats(left_linear_accs),
            "angular_acc": get_stats(left_angular_accs)
        },
        "right_stat": {
            "position": get_stats(right_positions),
            "orientation": get_stats_quat(right_quaternions),  # Use quaternion-aware stats
            "linear_speed": get_stats(right_linear_speeds),
            "angular_speed": get_stats(right_angular_speeds),
            "linear_acc": get_stats(right_linear_accs),
            "angular_acc": get_stats(right_angular_accs)
        },
        "duration": N * dt
    }
    if return_traj:
        results["left_positions"] = left_positions
        results["left_quaternions"] = left_quaternions
        results["right_positions"] = right_positions
        results["right_quaternions"] = right_quaternions
        
    return results

# Utility function: extract subtask segments from an episode
def get_subtask_index_range(dataset):
    subtask_segments = []
    
    current_subtask = None
    current_subtask_idx = None
    start_frame = None
    current_episode = None

    num_items = len(dataset)
    prev_frame = None

    for i in range(num_items):
        item = dataset.get_item_woimg(i)
        
        episode_idx = item['episode_index'].item()
        frame_idx = item['frame_index'].item()
        subtask_desc = item['subtasks'][0]
        subtask_idx = item['subtask_indices'][0].item()
        
        # Check if this is a valid subtask
        is_valid = (subtask_desc != "null")
        
        # If current frame belongs to a valid subtask
        if is_valid:
            # Start a new subtask if description or episode changes
            if (current_subtask is None 
                or current_subtask != subtask_desc 
                or current_episode != episode_idx):
                
                # Finalize previous subtask if exists
                if current_subtask is not None:
                    subtask_segments.append({
                        'ranges': {'start': start_frame, 'end': prev_frame},
                        'subtask_indice': current_subtask_idx,
                        'subtask': current_subtask
                    })
                
                # Initialize new subtask
                current_subtask = subtask_desc
                current_subtask_idx = subtask_idx
                current_episode = episode_idx
                start_frame = frame_idx
        else:
            # Current frame is "null": finalize ongoing subtask
            if current_subtask is not None:
                subtask_segments.append({
                    'ranges': {'start': start_frame, 'end': prev_frame},
                    'subtask_indice': current_subtask_idx,
                    'subtask': current_subtask
                })
                current_subtask = None
                current_subtask_idx = None
                start_frame = None
        
        prev_frame = frame_idx

    # Handle final unfinished subtask
    if current_subtask is not None:
        subtask_segments.append({
            'ranges': {'start': start_frame, 'end': prev_frame},
            'subtask_indice': current_subtask_idx,
            'subtask': current_subtask
        })
    
    return subtask_segments


        
def main():
    parser = argparse.ArgumentParser(description="Compute multi-dimensional statistics based on CoRobot-annotated subtasks and generate per-episode 'x-ray' summaries.")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    if not os.path.isdir(args.repo_path):
        raise ValueError(f"Path does not exist: {args.repo_path}")

    # Batch process: generate x-ray data for each episode
    repo_id = args.repo_path
    dataset = LeRobotDataset(repo_id, episodes=[0])
    num_episodes = len(dataset.meta.episodes)
    out_dir = os.path.join(repo_id, 'xray')
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_episodes):
        dataset = LeRobotDataset(repo_id, episodes=[i])
        subtask_segments = get_subtask_index_range(dataset)
        for subtask in subtask_segments:
            if subtask['ranges']['end'] - subtask['ranges']['start'] < 5: # Skip very short segments
                subtask['stats'] = {}
            else:
                subtask_stats = compute_hand_workspace_with_velocity_acceleration_from_qpos(
                    dataset, 
                    model,
                    ranges=[subtask['ranges']['start'], subtask['ranges']['end']],
                    dt=1/30
                )
                subtask['stats'] = subtask_stats
        with open(os.path.join(out_dir, 'episode_' + f"{i:06d}" + '.json'), 'w', encoding='utf-8') as f:
            json.dump(subtask_segments, f, ensure_ascii=False, indent=4)
            print(f"Episode {i} x-ray saved to {out_dir}")

if __name__ == "__main__":
    main()
