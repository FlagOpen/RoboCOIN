"""
Example command:

1. Dummy robot & dummy policy:

```python
python src/lerobot/scripts/replay.py \
    --robot.type=bi_realman     \
    --robot.ip_left="169.254.128.18"    \
    --robot.port_left=8080     \
    --robot.ip_right="169.254.128.19"     \
    --robot.port_right=8080     \
    --robot.block=True \
    --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}}"     \
    --robot.init_state="[-10.9, -123.7, 18.3, 37.8, 132.9, 101.4, -48.1, 718, 16.6, 116.3, -52.7,-21.2, -99.7, -84.2, 43.0, 956]"     \
    --robot.id=black  \
    --repo_id="realman/grasp_peach_new"
```
"""

import draccus
import traceback
from dataclasses import dataclass
from typing import Optional

import sys
sys.path.append('src/')

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.dummy.configuration_dummy import DummyCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots import (
    bi_piper,
    bi_realman_old,
    dummy,
    piper,
    realman_old,
)


@dataclass
class ReplayConfig:
    robot: RobotConfig
    repo_id: str
    video_backend: Optional[str] = None


class Replay:
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.dataset = LeRobotDataset(
            config.repo_id,
            video_backend=config.video_backend,
        )
    
    def start(self):
        self.robot.connect()

    def control_loop(self):
        for sample in self.dataset:
            self.robot.send_action(self._prepare_action(sample['action']))

    def stop(self):
        self.robot.disconnect()
    
    def _prepare_action(self, action: dict) -> dict:
        return {key: action[i].item() for i, key in enumerate(self.robot.action_features.keys())}


@draccus.wrap()
def main(cfg: ReplayConfig):
    replay = Replay(cfg)
    replay.start()
    
    try:
        replay.control_loop()
    except KeyboardInterrupt:
        replay.stop()
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
    finally:
        replay.stop()


if __name__ == "__main__":
    main()
