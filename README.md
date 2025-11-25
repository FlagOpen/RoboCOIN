# RoboCOIN

📄READMEs: [English](README.md) | [中文](README_zh-CN.md) | [LeRobot](README_lerobot.md)

🔗Links: [Project Website](https://flagopen.github.io/RoboCOIN/) | [ArXiv](https://arxiv.org/abs/2511.17441) | [PDF](https://arxiv.org/pdf/2511.17441) | [Visualize & Download](https://flagopen.github.io/RoboCOIN-DataManage/)

## News

- 🔥[24 Nov. 2025] Our technical report is available on [ArXiv](https://arxiv.org/abs/2511.17441)!

## Table of Contents
- [RoboCOIN](#robocoin)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dataset Discovery, Download, and Loading](#dataset-discovery-download-and-loading)
    - [Discover and Download Datasets](#discover-and-download-datasets)
    - [Load a Dataset](#load-a-dataset)
    - [Lerobot features explanation](#lerobot-features-explanation)
    - [Upcoming Highlights](#upcoming-highlights)
  - [Robot Control](#robot-control)
    - [Robot Script Structure](#robot-script-structure)
    - [Base Robot Configuration Classes](#base-robot-configuration-classes)
    - [Specific Robot Configuration Classes](#specific-robot-configuration-classes)
    - [Specific Feature Descriptions](#specific-feature-descriptions)
      - [Unified Unit Conversion](#unified-unit-conversion)
      - [Absolute and Relative Position Control](#absolute-and-relative-position-control)
    - [Usage Instructions](#usage-instructions)
      - [Trajectory Replay](#trajectory-replay)
      - [Model Inference](#model-inference)
        - [LeRobot Policy Based Inference](#lerobot-policy-based-inference)
        - [OpenPI Policy Based Inference](#openpi-policy-based-inference)
        - [Hierarchical Task Description Inference (Currently Only Supports OpenPI)](#hierarchical-task-description-inference-currently-only-supports-openpi)
    - [Customization](#customization)
      - [Adding Custom Robots](#adding-custom-robots)
  - [Acknowledgements](#acknowledgements)
  - [Contact Us](#contact-us)
## Overview

As the official companion toolkit for the [**RoboCOIN Dataset**], this project is built upon the [**LeRobot**](https://github.com/huggingface/lerobot) repository. It maintains full compatibility with LeRobot’s data format while adding support for rich metadata—including **subtasks**, **scene descriptions**, and **motion descriptions**. RoboCOIN provides an end-to-end pipeline for dataset discovery, download, and standardized loading, along with model deployment capabilities across multiple robotic platforms.

**Key Features**:
1. **Dataset Management**: Seamless retrieval, downloading, and `DataLoader`-based loading of datasets, with full support for subtask, scene, and motion annotation metadata.
2. **Unified Robot Control Interface**: Supports integration with diverse robotic platforms, including SDK-based control (e.g., Piper, Realman) and general-purpose ROS/MoveIt-based control.
3. **Standardized Unit Conversion**: Built-in utilities for cross-platform unit handling (e.g., degree ↔ radian conversion).
4. **Visualization Tools**: 2D/3D trajectory plotting and synchronized camera image rendering.
5. **Policy Inference & Deployment**: Ready-to-use inference pipelines for both **LeRobot Policy** and **OpenPI Policy**, enabling direct robot control from trained models.

---

## Installation

```bash
pip install robocoin
```

---
## Dataset Discovery, Download, and Loading

### Discover and Download Datasets  
> Browse available datasets at: [https://flagopen.github.io/RoboCOIN-DataManage/]
We will continuously update the datasets. You can find the latest datasets on the page above.


<p align="center">
  <img src="assets/how-to-use.gif" alt="This demo shows how to discovery, download, and standardized loading RoboCOIN datasets" width="700">
</p>
<p align="center">The above GIF shows how to discovery, download, and use RoboCOIN datasets.</p>


```bash
# you can copy the bash command from the website and paste it here, such as:
robocoin-download --hub huggingface --ds_lists Cobot_Magic_move_the_bread R1_Lite_open_and_close_microwave_oven

# the default download path is ~/.cache/huggingface/lerobot/, which will be used as default dir of LerobotDataset.
# if you want to speicifiy download dir, please add --target-dir YOUR_DOWNLOAD_DIR, such as:
# robocoin-download --hub huggingface --ds_lists Cobot_Magic_move_the_bread R1_Lite_open_and_close_microwave_oven --target-dir /path/to/your/download/dir

# we also provide a download option via ModelScope, such as:
# robocoin-download --hub modelscope --ds_lists Cobot_Magic_move_the_bread R1_Lite_open_and_close_microwave_oven 
```

### Load a Dataset
```python
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("RoboCOIN/R1_Lite_open_and_close_microwave_oven")

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=8,
    batch_size=32,
)
```
---
### Lerobot features explanation
#### `observation.state` / `action` 
These features represent data collected from the robot arms (slave/master). In the absence of robot action data, actions are derived from the observation.state sequence. The standardized fields are:

| Feature | Unit | Description |
|---|---:|---|
| `{dir}_arm_joint_{num}_rad` | rad | Converted from collected data; represents the arm joint angles (slave/master). | 
| `{dir}_hand_joint_{num}_rad` | rad | Converted from collected data; represents the hand joint angles. | 
| `{dir}_gripper_open` | - | Value range [0, 1]; `0` means fully closed, `1`means fully open; converted from collected data. | 
| `{dir}_eef_pos_{axis}` | m | EEF position obtained from robot sdk. |
| `{dir}_eef_rot_{axis}` | rad | EEF rotation(euler) obtained from robot sdk. |

#### `eef_sim_pose_state` / `eef_sim_pose_action`
These features represent simulation eef data. Due to inconsistencies in the coordinate system definitions across different robotic SDKs in the observation.state / action data, we employed a simulation-based approach to obtain the end-effector poses of each robot expressed in a unified coordinate system (x-forward / y-left / z-up, with the origin located at the robot's base or the center between its feet). These simulated end-effector poses are represented by the features `eef_sim_pose_state` / `eef_sim_pose_action`.

> Note: `{dir}` is a placeholder that stands for `left` or `right`.
---
### Upcoming Highlights

- **Version Compatibility**: RoboCOIN currently supports **LeRobot v2.1** data format. Support for **v3.0** data format is coming soon.
- **Codebase Origin**: This project is currently based on **LeRobot v0.3.4**. Future releases will evolve into a fully compatible **LeRobot extension plugin**, maintaining seamless interoperability with the official LeRobot repository.
---
## Robot Control

```mermaid
graph LR
    subgraph Robot Low-level Interfaces
    A1[Unified Unit Conversion]
    A2[Absolute & Relative Position Control]
    A3[Camera & Trajectory Visualization]
    A[Robot Low-level Interface]
    end
    
    %% Robot Service Layer
    subgraph Robot Services
    C[Robot Services]
    C1[SDK]
    C2[ROS]
    C11[Agilex Piper Service]
    C12[Realman Service]
    C13[Other Robot Services]
    C21[Generic Robot Service]
    end
    
    %% Camera Service Layer
    subgraph Camera Services
    D[Camera Services]
    D1[OpenCV Camera Service]
    D2[RealSense Camera Service]
    end
    
    %% Inference Service Layer
    subgraph Inference Services
    E[Inference Services]
    E1[RPC]
    E11[Lerobot Policy]
    E2[WebSocket]
    E21[OpenPi Policy]
    end
    
    %% Connection Relationships

    A1 --- A
    A2 --- A
    A3 --- A

    C --- C1
    C --- C2
    C1 --- C11
    C1 --- C12
    C1 --- C13
    C2 --- C21
    
    D --- D1
    D --- D2

    E --- E1
    E1 --- E11
    E --- E2
    E2 --- E21

    A --- C
    A --- D
    A --- E
    
```

### Robot Script Structure

All robot scripts are located under `src/lerobot/robots`. Taking the Realman robot platform as an example, all relevant files are located in src/lerobot/robots/realman(single arm) and `src/lerobot/robots/bi_realman`(dual arm):

```bash
realman # Single arm
├── __init__.py
├── configuration_realman.py # Configuration class
├── realman.py               # Joint control
└── realman_end_effector.py  # End effector control

bi_realman # Dual arm
├── __init__.py
├── bi_realman.py               # Joint control
├── bi_realman_end_effector.py  # End effector control
└── configuration_bi_realman.py # Configuration class
```

### Base Robot Configuration Classes

**Inheritance Relationship**：
```mermaid
graph LR
    A[RobotConfig] --> B[BaseRobotConfig]
    B --> C[BaseRobotEndEffectorConfig]
    B --> D[BiBaseRobotConfig]
    D --> E[BiBaseRobotEndEffectorConfig]
    C --> E
```

The base configuration for robot platforms is located at `src/lerobot/robots/base_robot/configuration_base_robot.py`：

```python
# Base configuration class for joint control
@RobotConfig.register_subclass("base_robot")
@dataclass
class BaseRobotConfig(RobotConfig):
    # Camera settings, represented as dictionary, key is camera name, value is camera config class, e.g.
    # {
    #     head: {type: opencv, index_or_path:0, height: 480, width: 640, fps: 30}, 
    #     wrist: {type: opencv, index_or_path:1, height: 480, width: 640, fps: 30},
    # }
    # The above example creates head and wrist cameras, loading /dev/video0, /dev/video1 respectively
    # Finally sent to model: {"observation.head": shape(480, 640, 3), "observation.wrist": shape(480, 640, 3)}
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Joint names, including gripper
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ]) 

    # Initialization mode: none for no initialization, joint/end_effector for joint/end effector based initialization
    init_type: str = 'none'
    # Values to initialize before starting inference based on initialization mode
    # For joint: units in radian
    # For end_effector: units in m (first 3 values) / radian (values 3~6)
    init_state: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

    # Joint control units, depends on SDK, e.g. Realman SDK has 7 joints receiving angles as parameters, should set:
    # ['degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm']
    # Last dimension is m, meaning gripper value doesn't need unit conversion
    joint_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    # End effector control units, depends on SDK, e.g. Realman SDK receives meters for xyz and degrees for rpy, should set:
    # ['m', 'm', 'm', 'degree', 'degree', 'degree', 'm']
    # Last dimension is m, meaning gripper value doesn't need unit conversion
    pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])
    # Model input joint control units, depends on dataset, e.g. if dataset saves in radians, should set:
    # ['radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm']
    # Last dimension is m, meaning gripper value doesn't need unit conversion
    model_joint_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    
    # Relative position control mode: none for absolute position control, previous/init for relative transformation based on previous/initial state
    # Taking joint control as example:
    # - If previous: obtained state + previous state -> target state
    # - If init: obtained state + initial state -> target state
    delta_with: str = 'none'

    # Whether to enable visualization
    visualize: bool = True
    # Whether to draw 2D trajectory, including end effector trajectory on XY, XZ, YZ planes
    draw_2d: bool = True
    # Whether to draw 3D trajectory
    draw_3d: bool = True


# Base configuration class for end effector control
@RobotConfig.register_subclass("base_robot_end_effector")
@dataclass
class BaseRobotEndEffectorConfig(BaseRobotConfig):
    # Relative transformation angles, applicable for cross-body scenarios where different bodies have different zero pose orientations
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Model input end effector control units, depends on dataset, e.g. if dataset saves in meters and radians, should set:
    # ['m', 'm', 'm', 'radian', 'radian', 'radian', 'm']
    # Last dimension is m, meaning gripper value doesn't need unit conversion
    model_pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])
```

Parameter Details：

| Parameter Name      | Type                      | Default Value                                                                              | Description                                                                                                                      |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `cameras`           | `dict[str, CameraConfig]` | `{}`                                                                                       | Camera configuration dictionary, key is camera name, value is camera configuration                                               |
| `joint_names`       | `List[str]`               | `['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper']` | Joint name list, including gripper                                                                                               |
| `init_type`         | `str`                     | `'none'`                                                                                   | Initialization type, options: `'none'`, `'joint'`, `'end_effector'`                                                              |
| `init_state`        | `List[float]`             | `[0, 0, 0, 0, 0, 0, 0, 0]`                                                                 | Initial state: joint state when `init_type='joint'`, end effector state when `init_type='end_effector'`                          |
| `joint_units`       | `List[str]`               | `['radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm']`              | Robot joint units, for SDK control                                                                                               |
| `pose_units`        | `List[str]`               | `['m', 'm', 'm', 'radian', 'radian', 'radian', 'm']`                                       | End effector pose units, for SDK control                                                                                         |
| `model_joint_units` | `List[str]`               | `['radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm']`              | Model joint units, for model input/output                                                                                        |
| `delta_with`        | `str`                     | `'none'`                                                                                   | Delta control mode: `'none'`(absolute control), `'previous'`(relative to previous state), `'initial'`(relative to initial state) |
| `visualize`         | `bool`                    | `True`                                                                                     | Whether to enable visualization                                                                                                  |
| `draw_2d`           | `bool`                    | `True`                                                                                     | Whether to draw 2D trajectory                                                                                                    |
| `draw_3d`           | `bool`                    | `True`                                                                                     | Whether to draw 3D trajectory                                                                                                    |

The dual-arm robot base configuration class is located at `src/lerobot/robots/base_robot/configuration_bi_base_robot.py`, inheriting from the single-arm base configuration:

```python
# Dual-arm robot configuration
@RobotConfig.register_subclass("bi_base_robot")
@dataclass
class BiBaseRobotConfig(BaseRobotConfig):
    # Left arm initial pose
    init_state_left: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])
    # Right arm initial pose
    init_state_right: List[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])


# Dual-arm robot end effector configuration
@RobotConfig.register_subclass("bi_base_robot_end_effector")
@dataclass
class BiBaseRobotEndEffectorConfig(BiBaseRobotConfig, BaseRobotEndEffectorConfig):
    pass
```

Parameter Details：

| Parameter Name     | Type          | Default Value              | Description                   |
| ------------------ | ------------- | -------------------------- | ----------------------------- |
| `init_state_left`  | `List[float]` | `[0, 0, 0, 0, 0, 0, 0, 0]` | Left arm initial joint state  |
| `init_state_right` | `List[float]` | `[0, 0, 0, 0, 0, 0, 0, 0]` | Right arm initial joint state |

### Specific Robot Configuration Classes

Each specific robot has dedicated configuration inheriting from the robot base configuration. Configure according to the specific robot SDK.

Inheritance relationship, taking Realman as example:

```mermaid
graph LR
    A[BaseRobotConfig] --> B[RealmanConfig]
    A --> C[RealmanEndEffectorConfig]
    D[BiBaseRobotConfig] --> E[BiRealmanConfig]
    D --> F[BiRealmanEndEffectorConfig]
    C --> F
```

Taking Realman as example, located at `src/lerobot/robots/realman/configuration_realman.py`：

```python
@RobotConfig.register_subclass("realman")
@dataclass
class RealmanConfig(BaseRobotConfig):
    ip: str = "169.254.128.18" # Realman SDK connection IP
    port: int = 8080           # Realman SDK connection port
    block: bool = False        # Whether to use blocking control
    wait_second: float = 0.1   # If non-blocking, delay after each action
    velocity: int = 30         # Movement velocity

    # Realman has 7 joints + gripper
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    # Use joint control to reach Realman's initial task pose
    init_type: str = "joint"
    init_state: list[float] = field(default_factory=lambda: [
        -0.84, -2.03,  1.15,  1.15,  2.71,  1.60, -2.99, 888.00,
    ])

    # Realman SDK defaults to meters + degrees
    joint_units: list[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


@RobotConfig.register_subclass("realman_end_effector")
@dataclass
class RealmanEndEffectorConfig(RealmanConfig, BaseRobotEndEffectorConfig):
    pass
```

For dual-arm Realman, configuration class is located at `src/lerobot/robots/bi_realman/configuration_bi_realman.py`：

```python
# Dual-arm Realman configuration
@RobotConfig.register_subclass("bi_realman")
@dataclass
class BiRealmanConfig(BiBaseRobotConfig):
    ip_left: str = "169.254.128.18" # Realman left arm SDK connection IP
    port_left: int = 8080 # Realman left arm SDK connection port
    ip_right: str = "169.254.128.19" # Realman right arm SDK connection IP
    port_right: int = 8080 # Realman right arm SDK connection port
    block: bool = False # Whether to use blocking control
    wait_second: float = 0.1 # If non-blocking, delay after each action
    velocity: int = 30 # Movement velocity
    
    # Realman has 7 joints + gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])
    
    # Use joint control to reach Realman's initial task pose
    init_type: str = "joint"
    init_state_left: List[float] = field(default_factory=lambda: [
        -0.84, -2.03,  1.15,  1.15,  2.71,  1.60, -2.99, 888.00,
    ])
    init_state_right: List[float] = field(default_factory=lambda: [
         1.16,  2.01, -0.79, -0.68, -2.84, -1.61,  2.37, 832.00,
    ])

    # Realman SDK defaults to meters + degrees
    joint_units: List[str] = field(default_factory=lambda: [
        'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm',
    ])
    pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'degree', 'degree', 'degree', 'm',
    ])


# Dual-arm Realman end effector configuration
@RobotConfig.register_subclass("bi_realman_end_effector")
@dataclass
class BiRealmanEndEffectorConfig(BiRealmanConfig, BiBaseRobotEndEffectorConfig):
    pass
```

### Specific Feature Descriptions

#### Unified Unit Conversion

This module is located at `src/lerobot/robots/base_robot/units_transform.py`, providing unit conversion functionality for length and angle measurements, supporting unified unit management in robot control systems: length uses meters (m), angles use radians (rad).

**​Length Unit Conversion**: Standard unit is meter (m), supports conversion between micrometer, millimeter, centimeter, meter.

| Unit       | Symbol     | Conversion Ratio |
| ---------- | ---------- | ---------------- |
| Micrometer | um (001mm) | 1 um = 1e-6 m    |
| Millimeter | mm         | 1 mm = 1e-3 m    |
| Centimeter | cm         | 1 cm = 1e-2 m    |
| Meter      | m          | 1 m = 1 m        |

**Angle Unit Conversion**: Standard unit is radian (rad), supports conversion between millidegree, degree, and radian.

| Unit        | Symbol        | Conversion Ratio     |
| ----------- | ------------- | -------------------- |
| Millidegree | mdeg (001deg) | 1 mdeg = π/18000 rad |
| Degree      | deg           | 1 deg = π/180 rad    |
| Radian      | rad           | 1 rad = 1 rad        |

During inference, the control units of the robot platform may differ from the model input/output units. This module provides unified conversion interfaces to ensure unit consistency and correctness during control:
1. Robot state to model input conversion: Robot specific units -> Standard units -> Model specific units
2. Model output to robot control conversion: Model specific units -> Standard units -> Robot specific units

```mermaid
sequenceDiagram
    participant A as Robot State (Specific Units)
    participant B as Standard Units
    participant C as Model Input/Output (Specific Units)
    A ->> B: 1. Convert to Standard Units
    B ->> C: 2. Convert to Model Specific Units
    C ->> B: 3. Convert to Standard Units
    B ->> A: 4. Convert to Robot Specific Units
```

#### Absolute and Relative Position Control

Provides three modes of position control: absolute, relative to previous state, and relative to initial state, applicable to both joint control and end effector control:
1. Absolute position control (absolute): Directly use model output position as target position
2. Relative to previous state position control (relative to previous): Use model output position as delta relative to previous state to calculate target position
   - Without action chunking: Action = Current state + Model output
   - With action chunking: Action = Current state + All model output chunks, update current state after all executions complete
3. Relative to initial state position control (relative to initial): Use model output position as delta relative to initial state to calculate target position

Example control flow using action chunking with relative to previous state position control:

```mermaid
sequenceDiagram
    participant Model as Model
    participant Controller as Controller
    participant Robot as Robot
    
    Note over Robot: Current State: st
    
    Model->>Controller: Output action sequence: [at+1, at+2, ..., at+n]
    
    Note over Controller: Actions always calculated relative to initial state st

    loop Execute action sequence i = 1 to n
        Controller->>Robot: Execute action: st + at+i
        Robot-->>Controller: Reach state st+i = st + at+i
    end
    
    Note over Robot: Final State: st+n
```


### Usage Instructions

#### Trajectory Replay

Robot platform configuration options can be modified in configuration class files or passed via command line. Taking dual-arm Realman as example, command is as follows:

```bash
python src/lerobot/scripts/replay.py \
    --repo_id=<your_lerobot_repo_id> \
    --robot.type=bi_realman \
    --robot.ip_left="169.254.128.18" \
    --robot.port_left=8080 \
    --robot.ip_right="169.254.128.19" \
    --robot.port_right=8080 \
    --robot.block=True \
    --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --robot.visualize=True
```

The above command specifies Realman left and right arm IP/ports, and loads head, left hand, right hand cameras. During trajectory replay, control will be based on data in <your_lerobot_repo_id>.

#### Model Inference

##### LeRobot Policy Based Inference

1. Run LeRobot Server, see `src/lerobot/scripts/server/policy_server.py`, command as follows:
```bash
python src/lerobot/scripts/server/policy_server.py \
    --host=127.0.0.1 \
    --port=18080 \
    --fps=10 
```
The above command starts a service listening on `127.0.0.1:18080`.

2. Run client program, taking dual-arm Realman as example, command as follows:
```bash
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=bi_realman \
    --robot.ip_left="169.254.128.18" \
    --robot.port_left=8080 \
    --robot.ip_right="169.254.128.19" \
    --robot.port_right=8080 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30},right_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30}}" \
    --robot.block=False \
    --robot.id=black \
    --fps=10 \
    --task="do something" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=path/to/checkpoint \
    --actions_per_chunk=50 \
    --verify_robot_cameras=False
```

The above command initializes Realman pose, loads head, left hand, right hand cameras, passes "do something" as prompt, loads ACT model for inference, and obtains actions to control the robot platform.

##### OpenPI Policy Based Inference

1. Run OpenPI Server, see [OpenPI official repository](https://github.com/Physical-Intelligence/openpi)
2. Run client program, taking Realman as example, command as follows:

```bash
python src/lerobot/scripts/server/robot_client_openpi.py \
  --host="127.0.0.1" \ # Server IP
  --port=8000 \ # Server port
  --task="put peach into basket" \ # Task instruction
  --robot.type=bi_realman \ # Realman configuration
  --robot.ip_left="169.254.128.18" \ 
  --robot.port_left=8080 \ 
  --robot.ip_right="169.254.128.19" \ 
  --robot.port_right=8080 \ 
  --robot.block=False \ 
  --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30}}" \ # 
  --robot.init_type="joint" \
  --robot.id=black
```

The above command initializes Realman pose, loads head, left hand, right hand cameras, passes "put peach into basket" as prompt, and obtains actions to control the robot platform.

During inference, press "q" in console to exit anytime, then press "y/n" to indicate task success/failure. Video will be saved to results/directory.

##### Hierarchical Task Description Inference (Currently Only Supports OpenPI)

First write a configuration class for the current task, e.g. `src/lerobot/scripts/server/task_configs/towel_basket.py`:

```python
@dataclass
class TaskConfig:
    # Scene description
    scene: str = "a yellow basket and a grey towel are place on a white table, the basket is on the left and the towel is on the right."
    # Task instruction
    task: str = "put the towel into the basket."
    # Subtask instructions
    subtasks: List[str] = field(default_factory=lambda: [
        "left gripper catch basket",
        "left gripper move basket to center",
        "right gripper catch towel",
        "right gripper move towel over basket and release",
        "end",
    ])
    # State statistics operators
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
```

Then run command:

```bash
python src/lerobot/scripts/server/robot_client_openpi_anno.py \
  --host="127.0.0.1" \
  --port=8000 \
  --task_config_path="lerobot/scripts/server/task_configs/towel_basket.py" \
  --robot.type=bi_realman_end_effector \
  --robot.ip_left="169.254.128.18" \
  --robot.port_left=8080 \
  --robot.ip_right="169.254.128.19" \
  --robot.port_right=8080 \
  --robot.block=False \
  --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30}}" \
  --robot.init_type="joint" \
  --robot.id=black
```

During inference, it starts from the first subtask, press "s" to switch to next subtask.

Press "q" in console to exit anytime, then press "y/n" to indicate task success/failure. Video will be saved to `results/` directory.

### Customization

#### Adding Custom Robots

1. Create a new folder under src/lerobot/robots/directory named after your robot, e.g. my_robot
2. Create the following files in this folder:
   - `__init__.py`: Initialization file
   - `my_robot.py`: Implement robot control logic
   - `configuration_my_robot.py`: Define robot configuration class, inheriting from RobotConfig
3. Define robot configuration in configuration_my_robot.py, including SDK-specific configuration and required base configuration parameters
4. Implement robot control logic in my_robot.py, inheriting from BaseRobot
5. Implement all abstract methods:
   - `_check_dependencys(self)`: Check robot dependencies
   - `_connect_arm(self)`: Connect to robot
   - `_disconnect_arm(self)`: Disconnect from robot
   - `_set_joint_state(self, joint_state: np.ndarray)`: Set robot joint state, input is joint state numpy array, units as defined in configuration class joint_units
   - `_get_joint_state(self) -> np.ndarray`: Get current robot joint state, returns joint state numpy array, units as defined in configuration class joint_units
   - `_set_ee_state(self, ee_state: np.ndarray)`: Set robot end effector state, input is end effector state numpy array, units as defined in configuration class pose_units
   - `_get_ee_state(self) -> np.ndarray`: Get current robot end effector state, returns end effector state numpy array, units as defined in configuration class pose_units
6. Refer to other robot implementation classes, implement other control modes (optional):
   - `my_robot_end_effector.py`: Implement end effector based control logic, inheriting from BaseRobotEndEffectorand my_robot.py
   - `bi_my_robot.py`: Implement dual-arm robot control logic, inheriting from BiBaseRobotand my_robot.py
   - `bi_my_robot_end_effector.py`: Implement dual-arm robot end effector based control logic, inheriting from BiBaseRobotEndEffectorand my_robot_end_effector.py
7. Register your robot configuration class in src/lerobot/robots/utils.py:
   ```python
   elif robot_type == "my_robot":
       from .my_robot.configuration_my_robot import MyRobotConfig
       return MyRobotConfig(**config_dict)
   elif robot_type == "my_robot_end_effector":
       from .my_robot.configuration_my_robot import MyRobotEndEffectorConfig
       return MyRobotEndEffectorConfig(**config_dict)
   elif robot_type == "bi_my_robot":
       from .my_robot.configuration_my_robot import BiMyRobotConfig
       return BiMyRobotConfig(**config_dict)
   elif robot_type == "bi_my_robot_end_effector":
       from .my_robot.configuration_my_robot import BiMyRobotEndEffectorConfig
       return BiMyRobotEndEffectorConfig(**config_dict)
   ```
8. Import your robot implementation class at the beginning of inference scripts:
   ```python
   from lerobot.robots.my_robot.my_robot import MyRobot
   from lerobot.robots.my_robot.my_robot_end_effector import MyRobotEndEffector
   from lerobot.robots.my_robot.bi_my_robot import BiMyRobot
   from lerobot.robots.my_robot.bi_my_robot_end_effector import BiMyRobotEndEffector
   ```
9. Now you can use your custom robot via command line parameter `--robot.type=my_robot`
---
## Acknowledgements

Thanks to the following open-source projects for their support and assistance to RoboCOIN:
- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)

---
## Contact Us
Welcome to scan the QR code to join the official RoboCOIN WeChat group for discussion.
![QR COde](https://flagopen.github.io/RoboCOIN/assets/qrcode-DFmJpXi2.png)
