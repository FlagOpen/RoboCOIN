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
  --robot.cameras="{ observation.images.cam_high: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, observation.images.cam_left_wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30},observation.images.cam_right_wrist: {type: opencv, index_or_path: 20, width: 640, height: 480, fps: 30}}"     \
  --robot.init_type="joint"     \
  --robot.init_state_left="[-0.77616538, -2.04976705,  1.6935104,   1.34390352, -0.24169319, -1.75644702,  0.86908667,  0.861     ]" \
  --robot.init_state_right="[ 0.91439543,  1.89743463, -1.03691755, -0.70560173, -2.48657061, -1.54884003, 1.88523482,  0.853     ]" \
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

        # for _ in range(100):
        #     obs = self._prepare_observation(self.robot.get_observation())

        while True:
            obs = self._prepare_observation(self.robot.get_observation())
            self.logger.info(f'Sent observation: {list(obs.keys())}')
            actions = self.policy.infer(obs)['action'][:32:4]
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

        observation['observation.state'] = state
        return observation
    
    def _prepare_action(self, action):
        assert len(action) == len(self.robot.action_features), \
            f"Action length {len(action)} does not match expected {len(self.robot.action_features)}: {self.robot.action_features.keys()}"
        action = np.array(action)

        # 判断gripper值是否小于600，如果是则设为20
        if action[7] > 1000:
            action[7] = 1000
        if action[7] < 300:
           action[7] = 0
        if action[-1] > 1000:
            action[-1] = 1000
        if action[-1] < 300:
           action[-1] = 0


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
