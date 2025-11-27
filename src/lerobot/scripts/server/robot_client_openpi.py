"""
Example command:

1. Dummy robot & dummy policy:

```python
python src/lerobot/scripts/server/robot_client_openpi.py \
    --host="127.0.0.1" \
    --port=18000 \
    --robot.type=bi_dummy_end_effector \
    --robot.cameras="{ observation.images.cam_high: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_left_wrist: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_right_wrist: {type: dummy, width: 640, height: 480, fps: 5}}" \
    --robot.id=black \
    --robot.delta_with="previous" \
    --robot.pose_units="[m, m, m, radian, radian, radian, m]" \
    --robot.model_pose_units="[m, m, m, radian, radian, radian, m]" \
    --task="fold the towel" \
    --fps 10
```

```python
python src/lerobot/scripts/server/robot_client_openpi.py \
    --host="127.0.0.1" \
    --port=18000 \
    --robot.type=bi_realman \
    --robot.cameras="{ observation.images.cam_high: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_left_wrist: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_right_wrist: {type: dummy, width: 640, height: 480, fps: 5}}" \
    --robot.init_type="joint"
    --robot.init_ee_state="[0, 0, 0, 0, 0, 0, 0]" \
    --robot.base_euler="[0, 0, 0]" \
    --robot.id=black 
```

peach
```python
python src/lerobot/scripts/server/robot_client_openpi.py \
  --host="127.0.0.1"     \
  --port=18000     \
  --task="put peach into basket"    \
  --robot.type=bi_realman_end_effector     \
  --robot.ip_left="169.254.128.18"    \
  --robot.port_left=8080     \
  --robot.ip_right="169.254.128.19"     \
  --robot.port_right=8080     \
  --robot.block=False \
  --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30}}"     \
  --robot.init_type="joint"     \
  --robot.id=black    \
  --robot.delta_with=previous
```

"""

import importlib

if importlib.util.find_spec("openpi_client") is None:
    raise ImportError("openpi_client is not installed. Please install it via `pip install openpi-client`.")

import draccus
import imageio
import numpy as np
import os
import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import List
from sshkeyboard import listen_keyboard, stop_listening

from openpi_client.websocket_client_policy import WebsocketClientPolicy

import sys
sys.path.append('src/')

from lerobot.cameras.dummy.configuration_dummy import DummyCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots import (
    bi_dummy,
    bi_piper,
    bi_realman,
    dummy,
    piper,
    realman,
)
from lerobot.scripts.server.helpers import get_logger


@dataclass
class OpenPIRobotClientConfig:
    robot: RobotConfig

    host: str = "127.0.0.1"
    port: int = 18000
    frequency: int = 10
    task: str = "do something"

    result_dir: str = "results/"
    camera_keys: List[str] = field(default_factory=lambda: [
        'observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist'
    ])
    fps: int = 10


class VideoRecorder:
    def __init__(
        self,
        save_dir,
        fps: int = 30,
    ):
        self.save_dir = save_dir
        self.fps = fps
        self._frames = []

        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, frame):
        if isinstance(frame, list):
            # [(H, W, C), ...] -> (H, W * N, C)
            frame = np.concatenate(frame, axis=1)
        self._frames.append(frame)
    
    def save(self, task, success):
        save_path = os.path.join(self.save_dir, f"{task.replace('.', '')}_{'success' if success else 'failed'}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        print(f'Saving video to {save_path}...')
        imageio.mimwrite(save_path, self._frames, fps=self.fps)
        self._frames = []


class KeyboardListener:
    def __init__(self):
        self._listener = threading.Thread(target=listen_keyboard, args=(self._on_press,))
        self._listener.daemon = True

        self._quit = False
        self._success = None
    
    def listen(self):
        self._listener.start()
    
    def reset(self):
        self._quit = False
        self._success = None
    
    def _on_press(self, key):
        if key == 'q':
            self._quit = True
        
        elif key == 'y':
            self._success = True
            stop_listening()
        
        elif key == 'n':
            self._success = False
            stop_listening()


class OpenPIRobotClient:
    def __init__(self, config: OpenPIRobotClientConfig):
        self.config = config
        self.logger = get_logger('openpi_robot_client')

        self.video_recorder = VideoRecorder(config.result_dir, fps=config.fps)
        self.keyboard_listener = KeyboardListener()

        self.policy = WebsocketClientPolicy(config.host, config.port)
        self.logger.info(f'Connected to OpenPI server at {config.host}:{config.port}')

        self.robot = make_robot_from_config(config.robot)
        self.logger.info(f'Initialized robot: {self.robot.name}')

        self._is_finished = False
    
    def start(self):
        self.keyboard_listener.listen()
        self.logger.info('Starting robot client...')
        self.robot.connect()
    
    def control_loop(self):
        while not self._is_finished:
            obs = self._prepare_observation(self.robot.get_observation())
            # self.logger.info(f'Sent observation: {list(obs.keys())}')
            self.logger.info(f'Prompt: {obs["prompt"]}')
            actions = self.policy.infer(obs)['action']
            for action in actions:
                action = self._prepare_action(action)
                self.logger.info(f'Received action: {action}')
                self.robot.send_action(action)
                self._after_action()
            time.sleep(1 / self.config.frequency)

    def stop(self):
        self.logger.info('Stopping robot client...')
        self.robot.disconnect()
    
    def _prepare_observation(self, observation):
        state = []
        for key in self.robot._motors_ft.keys():
            assert key in observation, f"Expected key {key} in observation, but got {observation.keys()}"
            state.append(observation[key])
            observation.pop(key)
        
        state = np.array(state)

        observation['observation.state'] = state
        observation['prompt'] = self.config.task
        return observation
    
    def _prepare_action(self, action):
        assert len(action) == len(self.robot.action_features), \
            f"Action length {len(action)} does not match expected {len(self.robot.action_features)}: {self.robot.action_features.keys()}"
        action = np.array(action)

        # 判断gripper值是否小于600，如果是则设为20
        if action[6] > 1000:
            action[6] = 1000
        if action[6] < 300:
           action[6] = 0
        if action[-1] > 1000:
            action[-1] = 1000
        if action[-1] < 300:
           action[-1] = 0

        return {key: action[i].item() for i, key in enumerate(self.robot.action_features.keys())}

    def _after_action(self):
        obs = self.robot.get_observation()
        frames = [obs[key] for key in self.config.camera_keys]
        self.video_recorder.add(frames)

        if self.keyboard_listener._quit:
            print('Success? (y/n): ', end='', flush=True)
            while self.keyboard_listener._success is None:
                time.sleep(0.1)
            print('Got:', self.keyboard_listener._success)
            self.video_recorder.save(task=self.config.task, success=self.keyboard_listener._success)
            self._is_finished = True


@draccus.wrap()
def main(cfg: OpenPIRobotClientConfig):
    client = OpenPIRobotClient(cfg)
    client.start()

    try:
        client.control_loop()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        client.logger.error(f'Error in control loop: {e}')
        client.logger.error(traceback.format_exc())
    finally:
        client.stop()


if __name__ == "__main__":
    main()
