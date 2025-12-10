import draccus
import imageio
import numpy as np
import os
import threading
import time
import torch
import traceback
from dataclasses import dataclass, field
from sshkeyboard import listen_keyboard, stop_listening
from typing import List

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.server.helpers import (
    map_robot_keys_to_lerobot_features,
    raw_observation_to_observation,
)
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.cameras.dummy.configuration_dummy import DummyCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots import (
    dummy,
    bi_dummy,
    piper,
    bi_piper,
    realman,
    bi_realman,
)


@dataclass
class LocalRobotClientConfig:
    robot: RobotConfig
    task: str = "place the bowl on the plate"
    pretrained_path: str = ""
    device: str = "cuda:0"
    repo_id: str = "piper/place_the_bowl_on_the_plate_filtered"
    result_dir: str = "results/"
    frequency: int = 30
    camera_keys: List[str] = field(default_factory=lambda: [
        'front'
    ])
    

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


class LocalRobotClient:
    def __init__(self, config: LocalRobotClientConfig):
        self.config = config

        self.video_recorder = VideoRecorder(config.result_dir, fps=config.frequency)
        self.keyboard_listener = KeyboardListener()

        self.dataset = LeRobotDataset(repo_id=config.repo_id)

        policy_config = PreTrainedConfig.from_pretrained(self.config.pretrained_path)
        policy_config.pretrained_path = self.config.pretrained_path
        self.policy = make_policy(policy_config, ds_meta=self.dataset.meta)
        self.policy.to(config.device)

        self.robot = make_robot_from_config(config.robot)

        self._is_finished = False
    
    def start(self):
        self.keyboard_listener.listen()
        self.robot.connect()
        time.sleep(5)
    
    def control_loop(self):
        while not self._is_finished:
            obs = self._prepare_observation(self.robot.get_observation())
            with torch.inference_mode():
                action = self.policy.select_action(obs)[0]
            obs = self.robot.get_observation()
            state = None
            action = self._prepare_action(action, state)
            print('Prepared action:', action)
            self.robot.send_action(action)
            self._after_action()
            time.sleep(1 / self.config.frequency)

    def stop(self):
        self.robot.disconnect()
    
    def _prepare_observation(self, observation):
        observation['task'] = self.config.task
        observation = raw_observation_to_observation(
            observation, 
            map_robot_keys_to_lerobot_features(self.robot),
            self.policy.config.image_features,
            device=self.config.device,
        )
        return observation
    
    def _prepare_action(self, action, state):
        return {k: action[i].item() for i, k in enumerate(self.robot.action_features.keys())}

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
def main(cfg: LocalRobotClientConfig):
    client = LocalRobotClient(cfg)
    client.start()

    try:
        client.control_loop()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        traceback.print_exc()
    finally:
        client.stop()


if __name__ == "__main__":
    main()