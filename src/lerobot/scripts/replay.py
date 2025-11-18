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
    --robot.block=False \
    --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}}"     \
    --robot.id=black  \
    --robot.visualize=True \
    --repo_id=realman/grasp_peach_new 
```
"""

import draccus
import numpy as np
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
    bi_dummy,
    bi_piper,
    bi_realman,
    dummy,
    piper,
    realman,
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
    
    def _prepare_action(self, action) -> dict:
        def rotation_6d_to_rpy(rotation_6d):
            r1 = rotation_6d[0:3]
            r2 = rotation_6d[3:6]
            r3 = np.cross(r1, r2)
            r_mat = np.stack([r1, r2, r3], axis=-1)
            sy = np.sqrt(r_mat[0, 0] * r_mat[0, 0] + r_mat[1, 0] * r_mat[1, 0])
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(r_mat[2, 1], r_mat[2, 2])
                y = np.arctan2(-r_mat[2, 0], sy)
                z = np.arctan2(r_mat[1, 0], r_mat[0, 0])
            else:
                x = np.arctan2(-r_mat[1, 2], r_mat[1, 1])
                y = np.arctan2(-r_mat[2, 0], sy)
                z = 0
            return np.array([x, y, z])

        left_gripper = action[7:8]
        left_xyz = action[8:11]
        left_rpy = action[11:17]
        right_gripper = action[24:25]
        right_xyz = action[25:28]
        right_rpy = action[28:34]
        left_rpy = rotation_6d_to_rpy(left_rpy)
        right_rpy = rotation_6d_to_rpy(right_rpy)
        action = np.concatenate([
            left_xyz, left_rpy, left_gripper, 
            right_xyz, right_rpy, right_gripper
        ], axis=0)
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