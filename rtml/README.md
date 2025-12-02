

## 🤖 Robot Data Clinic (RDC)

To address the lack of a unified quality evaluation standard for robot manipulation data, we propose **Robot Trajectory Markup Language (RTML)** — a domain-specific language (DSL) designed for trajectory quality assessment.

RTML translates empirical quality dimensions (e.g., smoothness, stability, coordination) into **computable, configurable, and machine-readable rules**, supporting both **global constraints** and **subtask-level customization**.

Built upon RTML, we introduce **Robot Data Clinic (RDC)** — an end-to-end system for trajectory data evaluation and enhancement, forming a closed-loop pipeline: **Diagnose → Filter → Enhance**.

<img src="assets/robot_data_clinic.png" width="800" alt="robot_data_clinic">


### 🔍 1. Trajectory Analyser

- Takes raw trajectory data as input and outputs multidimensional statistical features (pose, velocity, acceleration, time, etc.)
- Generates a "**Robot Data X-Ray**", visualizing spatial distributions, orientation anomalies, and other patterns of the end-effector
- Establishes an objective quality baseline for subsequent filtering

### ✅ 2. Trajectory Selector

- Automatically filters low-quality trajectories using an RTML rule library
- Supports two levels of constraints:
  - **Global constraints**: Basic quality requirements applicable to all tasks (e.g., max velocity, acceleration)
  - **Subtask constraints**: Fine-grained rules tailored to specific operation phases (e.g., workspace boundaries, idle-arm motion limits)
- Filters rigorously across five quality dimensions:
  - Motion smoothness  
  - Pose stability  
  - Bimanual coordination  
  - Task focus  
  - Execution efficiency  

---

## 🚀 Quick Start

### Environment Setup

```bash
conda create -n rdc python=3.10
conda activate rdc
conda install pinocchio -c conda-forge
pip install -r requirements.txt
```

---

### 📊 robot_trajectory_analyser

#### Generate X-Ray Statistics

Compute multidimensional statistics based on CoRobot-annotated subtasks and generate per-episode X-Ray files:

```bash
cd RTML_release/src/robot_trajectory_analyser
python compute_subtask_stats.py --repo_path /media/woxue/garbase/eai_datasets/lerobot/unitree_G1_BAAI_v2/unitree_g1_plate_storage_doll
```

> 💡 You can also interactively debug components and visualize end-effector pose changes and spatial trajectories in the notebook: `compute_subtask_stats.ipynb`.

| End-Effector Pose Distribution (Spherical Plot) | End-Effector Spatial Trajectory |
|------------------------------------------------|----------------------------------|
| <img src="assets/vis_orientation.png" width="500">  | <img src="assets/vis_trajectory.png" width="220"> |

#### Multidimensional Visualization

Visualize a specific statistic for a given subtask:

```bash
cd RTML_release/src/robot_trajectory_analyser
python visualize_subtask_stats.py \
  --subtask "Grasp the long bread with left hand" \
  --path /path/to/xray \
  --field left_stat.orientation.std \
  --output results/xray_vis.png
```

Batch-generate visualizations for all subtasks and statistics:

```bash
cd RTML_release/src/robot_trajectory_analyser
python __scripts/visualize_subtask_batch.py \
  --dataset_path /path/to/xray \
  --output_root results/xray_vis_all
```

> Example outputs:
>
> | Workspace Dimension | Velocity Dimension |
> |---------------------|--------------------|
> | ![Workspace](assets/statistics_workspace.png) | ![Velocity](assets/statistics_velocity.png) |

---

### 🧪 robot_trajectory_selector

#### Write an RTML File

Craft RTML rules based on statistical insights and subtask characteristics.

- Define **global constraints** (task-agnostic quality baselines):

```yaml
global_constraints:
  velocity:
    linear:
      max: 0.5
      mean_max: 0.3
  acceleration:
    linear:
      max: 12.0
```

- Define **stage-specific constraints** per subtask (e.g., workspace bounds, idle-arm limits):

```yaml
stages:
  - id: "move_bowl_right"
    match_subtask: "Move the pink bowl to the center of table with right hand"
    constraints:
      workspace:
        right:
          min: [0.10, -0.40, 0.10]
          max: [0.25, -0.20, 0.30]
      velocity:
        linear:
          mean_max: 0.10
          std_max: 0.08
      idle_arm:
        arm: "left"
        velocity_linear_mean_max: 0.05
      temporal:
        duration_min: 2.0
        duration_max: 6.0
```

> 📄 Full example: `RTML_release/src/robot_trajectory_selector/RTML_demo.yaml`

#### Filter High-Quality Episodes

Apply RTML rules to filter high-quality trajectories:

```bash
python robot_trajectory_selector.py \
  --input /mnt/phecda/woxue/projects/RobotDataFactory/RTML_release/sample_data/xray \
  --rtml RTML_demo.yaml \
  --output_file ../../results/hq_episodes.txt
```
