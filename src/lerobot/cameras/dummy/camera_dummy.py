import numpy as np
from typing import Any

from lerobot.cameras.camera import Camera


class DummyCamera(Camera):
    """
    Dummy camera implementation for testing purposes.
    This camera returns random rgb images instead of capturing from hardware.

    Example:
        ```python
        config = DummyCameraConfig(fps=30, width=640, height=480)
        camera = DummyCamera(config)
        camera.connect()

        # frame: np.ndrray of shape (height, width, 3) with random values
        frame = camera.read()

        camera.disconnect()
        ```
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._is_connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        print("Dummy camera connected")
        self._is_connected = True
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("DummyCamera does not support method find_cameras")

    def read(self) -> np.ndarray:
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        print("Dummy camera disconnected")
        self._is_connected = False

