# RoboCoin-LeRobot说明文档

## 概述

RoboCoin-LeRobot是一个基于LeRobot扩展的机器人部署环境，主要修改为：

1. 新增与真实机器人平台的交互逻辑：Agilex Piper、Realman，以及虚拟机器人平台的交互逻辑
2. 优化机器人控制逻辑：如单位转换、绝对与相对位置控制、关节与末端控制、相机与轨迹可视化等
3. 优化客户端推理逻辑：加入键盘交互、视频记录等功能
4. 提供与$\pi_0$服务端交互的客户端部署工具
5. 提供扩展机器人平台的良好封装

## 安装

```bash
pip install -e .
pip install -e third_party/openpi-client
```

## 了解机器人平台的控制逻辑

所有机器人平台都在`src/lerobot/robots`下，以Realman机器人平台为例，相应的所有文件位于`src/lerobot/robots/realman`（单臂）与`src/lerobot/robots/bi_realman`（双臂）下:

```bash
realman # 单臂
├── __init__.py
├── configuration_realman.py # 配置类
├── realman.py               # 关节控制
└── realman_end_effector.py  # 末端控制

bi_realman # 双臂
├── __init__.py
├── bi_realman.py               # 关节控制
├── bi_realman_end_effector.py  # 末端控制
└── configuration_bi_realman.py # 配置类
```

机器人平台的基础配置位于`src/lerobot/robots/base_robot/configuration_base_robot.py`：

```python
# 关节控制的基础配置类
@RobotConfig.register_subclass("base_robot")
@dataclass
class BaseRobotConfig(RobotConfig):
    # 相机设置，表示为字典，字典key为相机名，value为相机配置类，如
    # {
    #     head: {type: opencv, index_or_path:0, height: 480, width: 640, fps: 30}, 
    #     wrist: {type: opencv, index_or_path:1, height: 480, width: 640, fps: 30},
    # }
    # 上述示例创建了head和wrist两个相机，分别加载了/dev/video0, /dev/video1
    # 最终发送给模型的将是{"observation.head": shape(480, 640, 3), "observation.wrist": shape(480, 640, 3)}
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # 关节名称，包含夹爪
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ]) 

    # 初始化模式：none表示不进行初始化，joint/end_effector表示基于关节/末端初始化
    init_type: str = 'none'
    # 根据初始化模式，在开始推理之前要初始化的值
    # 对于joint，单位为radian
    # 对于end_effector，单位为m(前3个值) / radian（3~6个值）
    init_state: list[float] = field(default_factory=lambda: [
        0, 0, 0, 0, 0, 0, 0, 0,
    ])

    # 各关节控制单位，视SDK而定，如Realman SDK共7个关节，接收角度作为参数，则应设为：
    # ['degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'degree', 'm']
    # 最后一维为m，表示夹爪值不用进行单位转换
    joint_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    # 末端控制单位，视SDK而定，如Realman SDK接收米作为xyz和角度作为rpy，则应设为：
    # ['m', 'm', 'm', 'degree', 'degree', 'degree', 'm']
    # 最后一维为m，表示夹爪值不用进行单位转换
    pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])
    # 模型接收的关节控制单位，视数据集而定，如数据集中保存的单位为弧度，则应设为：
    # ['radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm']
    # 最后一维为m，表示夹爪值不用进行单位转换
    model_joint_units: list[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    
    # 相对位置控制模式：none表示绝对位置控制，previous/init表示基于上一状态或初始状态进行相对转换
    # 以关节控制为例:
    # - 若为previous：则得到的state + 上一个state -> 要达到的state
    # - 若为init：则得到的state + 初始state -> 要达到的state
    delta_with: str = 'none'

    # 是否启用可视化
    visualize: bool = True
    # 是否绘制2D轨迹图，包含XY, XZ, YZ平面上的末端轨迹
    draw_2d: bool = True
    # 是否绘制3D轨迹图
    draw_3d: bool = True


# 末端控制的基础配置类
@RobotConfig.register_subclass("base_robot_end_effector")
@dataclass
class BaseRobotEndEffectorConfig(BaseRobotConfig):
    # 相对变换角，适用于跨本体的情况，即不同本体的零姿态具有不同的朝向，则需要通过该参数进行变换
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 模型接收的末端控制单位，视数据集而定，如数据集中保存的单位是米和弧度，则应设为：
    # ['m', 'm', 'm', 'radian', 'radian', 'radian', 'm']
    # 最后一维为m，表示夹爪值不用进行单位转换
    model_pose_units: list[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])
```

每个机器人平台都有专门的扩展配置，以realman为例：

```python
@RobotConfig.register_subclass("realman")
@dataclass
class RealmanConfig(BaseRobotConfig):
    ip: str = "169.254.128.18" # Realman SDK连接ip
    port: int = 8080           # Realman SDK连接端口
    block: bool = False        # 是否阻塞控制
    wait_second: float = 0.1   # 如果非阻塞，每次行动后延迟多久
    velocity: int = 30         # 移动速度

    # Realman共有7个关节 + 夹爪
    joint_names: list[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])

    # 使用joint控制达到Realman执行任务的初始姿态
    init_type: str = "joint"
    init_state: list[float] = field(default_factory=lambda: [
        -0.84, -2.03,  1.15,  1.15,  2.71,  1.60, -2.99, 888.00,
    ])

    # Realman SDK默认采用米 + 角度
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

## 轨迹重播

机器人平台的配置选项可以在配置类文件中修改，也可以通过命令行传入，以双臂Realman为例，命令如下：

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

上述命令指定了Realman左臂与右臂的IP/端口，提供了相机索引。该命令将打开一个可视化窗口，从中可以查看相机图像与轨迹。

## 模型推理

**基于LeRobot Policy的推理**

TODO

**基于OpenPI Policy的推理**

1. 运行OpenPI Server
2. 运行客户端程序，以Realman为例，命令如下：

```bash
python src/lerobot/scripts/server/robot_client_openpi.py \
  --host="127.0.0.1" \ # 服务端IP
  --port=8000 \ # 服务端端口号
  --task="put peach into basket" \ # 任务指令
  --robot.type=bi_realman \ # Realman的配置项
  --robot.ip_left="169.254.128.18" \ 
  --robot.port_left=8080 \ 
  --robot.ip_right="169.254.128.19" \ 
  --robot.port_right=8080 \ 
  --robot.block=False \ 
  --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30}}" \ # 
  --robot.init_type="joint" \
  --robot.id=black
```

上述命令将初始化realman姿态，加载头部、左手、右手相机，传入"put peach into basket"作为prompt，并获取action对机器人平台进行控制。

推理时，可以在控制台中按"q"随时退出，之后按"y/n"表示当前任务成功或失败，视频将被存放到`results/`目录中。

**层次化任务描述的推理 (目前仅支持OpenPI)**

首先为当前任务编写一个配置类，如`src/lerobot/scripts/server/task_configs/towel_basket.py`:

```python
@dataclass
class TaskConfig:
    # 场景描述
    scene: str = "a yellow basket and a grey towel are place on a white table, the basket is on the left and the towel is on the right."
    # 任务指令
    task: str = "put the towel into the basket."
    # 子任务指令
    subtasks: List[str] = field(default_factory=lambda: [
        "left gripper catch basket",
        "left gripper move basket to center",
        "right gripper catch towel",
        "right gripper move towel over basket and release",
        "end",
    ])
    # 状态统计算子
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

之后运行命令：

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

推理时，将从第一个子任务开始，按"s"切换到下一个子任务。
可以在控制台中按"q"随时退出，之后按"y/n"表示当前任务成功或失败，视频将被存放到`results/`目录中。

## 新增自定义机器人

TODO