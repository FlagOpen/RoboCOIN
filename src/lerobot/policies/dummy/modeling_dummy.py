import builtins
import numpy as np
import torch
from collections import deque
from pathlib import Path
from torch import Tensor
from typing import TypeVar

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_dummy import DummyConfig

T = TypeVar("T", bound="PreTrainedPolicy")


class DummyPolicy(PreTrainedPolicy):
    """
    DummyPolicy is not a real policy with learning capabilities.
    It is a placeholder that returns a fixed action.
    It is used for testing and development purposes, simulating a policy that always outputs the same action.

    Example:
        ```python
        policy = DummyPolicy.from_pretrained("[0.1, 0, 0, 0, 0.1, 0, 0]")
        # actions: (Tensor) with shape (batch_size, chunk_size, action_len)
        actions = policy.predict_action_chunk(batch)
        ```
    """

    config_class = DummyConfig
    name = "dummy"

    def __init__(
        self,
        config: DummyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.num_action_steps = config.num_action_steps
        if isinstance(config.action, str):
            actions = np.load(config.action, allow_pickle=True)
            self.actions = torch.from_numpy(actions)
        else:
            self.actions = torch.tensor(config.action, dtype=torch.float32)
        self.index = 0
        self.reset()
    
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        from_pretrained method of DummyPolicy does not load a pretrained model,
        but rather returns a new instance of DummyPolicy with the provided configuration.
        """

        print('DummyPolicy does not need method from_pretrained, return a new instance directly.')
        try:
            if config is None:
                config = DummyConfig(
                    input_features={
                        "observation.images.front": PolicyFeature(
                            type=FeatureType.VISUAL,
                            shape=(3, 480, 640),
                        ),
                        "observation.images.front_fisheye": PolicyFeature(
                            type=FeatureType.VISUAL,
                            shape=(3, 480, 640),
                        ),
                        "observation.images.left_wrist": PolicyFeature(
                            type=FeatureType.VISUAL,
                            shape=(3, 480, 640),
                        ),
                        "observation.images.left_wrist_fisheye": PolicyFeature(
                            type=FeatureType.VISUAL,
                            shape=(3, 480, 640),
                        ),
                        "observation.images.right_wrist": PolicyFeature(
                            type=FeatureType.VISUAL,
                            shape=(3, 480, 640),
                        ),
                        "observation.images.right_wrist_fisheye": PolicyFeature(
                            type=FeatureType.VISUAL,
                            shape=(3, 480, 640),
                        ),
                    },
                    output_features={
                        "action": PolicyFeature(
                            type=FeatureType.ACTION,
                            shape=(7,),
                        ),
                    }
                )
            if pretrained_name_or_path.startswith('[') and pretrained_name_or_path.endswith(']'):
                print(pretrained_name_or_path)
                actions = [float(x) for x in pretrained_name_or_path[1:-1].split(',')]
                config.action = actions
            else:
                config.action = pretrained_name_or_path
            policy = DummyPolicy(config, **kwargs)
        except:
            import traceback
            print(traceback.format_exc())
        return policy

    def get_optim_params(self):
        return {}
    
    def reset(self):
        self._action_queue = deque([], maxlen=self.num_action_steps)
        self.index = 0
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, :, self.num_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        # (7,) -> (1, 1, 7) -> (B, N, 7)
        if self.actions.dim() == 1:
            actions = self.actions.unsqueeze(0).unsqueeze(0).repeat(
                1, self.num_action_steps, 1
            )
        else:
            if self.index < self.actions.shape[0]:
                actions = self.actions[self.index].unsqueeze(0).unsqueeze(0).repeat(
                    1, self.num_action_steps, 1
                )
                self.index += 10
            else:
                actions = self.actions[-1].unsqueeze(0).unsqueeze(0).repeat(
                    1, self.num_action_steps, 1
                )
        return actions

    def forward(self, batch):
        pass

    def push_model_to_hub(self, cfg):
        pass

    def generate_model_card(self, dataset_repo_id, model_type, license, tags):
        pass
