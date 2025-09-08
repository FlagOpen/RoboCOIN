"""
Example command:

1. Dummy robot & dummy policy:

```python
python src/lerobot/scripts/server/robot_client_openpi.py \
    --host="127.0.0.1" \
    --port=18000 \
    --robot.type=dummy \
    --robot.control_mode=ee_delta_gripper \
    --robot.cameras="{ observation.images.cam_high: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_left_wrist: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_right_wrist: {type: dummy, width: 640, height: 480, fps: 5}}" \
    --robot.init_ee_state="[0, 0, 0, 0, 0, 0, 0]" \
    --robot.base_euler="[0, 0, 0]" \
    --robot.id=black 
```

```python
python src/lerobot/scripts/server/robot_client_openpi.py \
    --host="127.0.0.1" \
    --port=18000 \
    --robot.type=realman \
    --robot.cameras="{ observation.images.cam_high: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_left_wrist: {type: dummy, width: 640, height: 480, fps: 5},observation.images.cam_right_wrist: {type: dummy, width: 640, height: 480, fps: 5}}" \
    --robot.init_ee_state="[0, 0, 0, 0, 0, 0, 0]" \
    --robot.base_euler="[0, 0, 0]" \
    --robot.id=black 
```

peach
```python
python src/lerobot/scripts/server/robot_client_openpi.py \
--host="172.16.19.138"     \
--port=18000     \
--robot.type=bi_realman     \
--robot.ip_left="169.254.128.18"    \
--robot.port_left=8080     \
--robot.ip_right="169.254.128.19"     \
--robot.port_right=8080     \
--robot.block=False \
--robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}}"     \
--robot.init_state="[-10.9, -123.7, 18.3, 37.8, 132.9, 101.4, -48.1, 718, 16.6, 116.3, -52.7,-21.2, -99.7, -84.2, 43.0, 956]"     \
--robot.id=black 
```

banana
```python
python src/lerobot/scripts/server/robot_client_openpi.py \
--host="172.16.19.138"     \
--port=18000     \
--robot.type=bi_realman     \
--robot.ip_left="169.254.128.18"    \
--robot.port_left=8080     \
--robot.ip_right="169.254.128.19"     \
--robot.port_right=8080     \
--robot.block=False \
--robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}}"     \
--robot.init_state="[-14.8, -116.3, -79.6, -58.2, 55.7, -13.9, 97.2, 910, 14.3, 114.6, 79.0, 55.9, -55.6, 14.9, -97.5, 904]"    \
--robot.id=black 
```

"""

import draccus
import math
import numpy as np
import time
import traceback
from dataclasses import dataclass

from openpi_client.websocket_client_policy import WebsocketClientPolicy

import sys
sys.path.append('src/')

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
from lerobot.scripts.server.helpers import get_logger


@dataclass
class OpenPIRobotClientConfig:
    robot: RobotConfig
    host: str = "127.0.0.1"
    port: int = 18000
    frequency: int = 10
    prompt: str = "do something"


class OpenPIRobotClient:
    def __init__(self, config: OpenPIRobotClientConfig):
        self.config = config
        self.logger = get_logger('openpi_robot_client')

        self.policy = WebsocketClientPolicy(config.host, config.port)
        self.logger.info(f'Connected to OpenPI server at {config.host}:{config.port}')

        self.robot = make_robot_from_config(config.robot)
        self.logger.info(f'Initialized robot: {self.robot.name}')
    
    def start(self):
        self.logger.info('Starting robot client...')
        self.robot.connect()
    
    def control_loop(self):
        # signal.signal(signal.SIGINT, quit)                                
        # signal.signal(signal.SIGTERM, quit)

        for _ in range(100):
            obs = self._prepare_observation(self.robot.get_observation())

        while True:
            obs = self._prepare_observation(self.robot.get_observation())
            self.logger.info(f'Sent observation: {list(obs.keys())}')
            actions = self.policy.infer(obs)['action'] #[:16]
            # actions = actions[:32:1]
            # actions = [actions[31]]
            for action in actions:
                action = self._prepare_action(action)
                self.logger.info(f'Received action: {action}')
                self.robot.send_action(action)
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

        state[:7] *= math.pi / 180
        state[8:-1] *= math.pi / 180

        observation['observation.state'] = state
        from PIL import Image
        Image.fromarray(observation['observation.images.cam_high']).save('test.jpg')
        # exit()
        return observation
    
    def _prepare_action(self, action):
        assert len(action) == len(self.robot.action_features), \
            f"Action length {len(action)} does not match expected {len(self.robot.action_features)}: {self.robot.action_features.keys()}"
        action = np.array(action)
        action[:7] *= 180 / math.pi
        action[8:-1] *= 180 / math.pi

    # 判断gripper值是否小于600，如果是则设为20
        if action[7] > 1000:
            action[7] = 1000
        if action[7] < 600:
           action[7] = 1
        if action[-1] > 1000:
            action[-1] = 1000
        if action[-1] < 600:
           action[-1] = 1


        return {key: action[i].item() for i, key in enumerate(self.robot.action_features.keys())}


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
