import logging
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class ActLeRobotCore:
    """ACT inference wrapper.

    The production path loads an ONNX file exported by export_act_to_onnx.py.
    A LeRobot directory fallback is kept for host-side debugging only.
    """

    def __init__(
        self,
        policy_path: str,
        dataset_repo_id: str = "",
        dataset_root: str = "",
        device: str = "cuda",
        image_height: int = 200,
        image_width: int = 320,
        crop_top_ratio: float = 0.375,
        crop_bottom_ratio: float = 0.0,
        state_mode: str = "vehicle_status",
        control_mode: str = "ai",
        acceleration: float = 0.6,
        n_action_steps: int = 20,
    ):
        if not policy_path:
            raise ValueError("model.policy_path is required")
        if crop_top_ratio + crop_bottom_ratio >= 1.0:
            raise ValueError("crop_top_ratio + crop_bottom_ratio must be < 1.0")

        self.logger = logging.getLogger(__name__)
        self.policy_path = Path(policy_path)
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.state_mode = state_mode
        self.control_mode = control_mode.lower()
        self.acceleration = acceleration
        self.n_action_steps = n_action_steps
        self._action_queue: deque[np.ndarray] = deque(maxlen=max(1, n_action_steps))

        if self.policy_path.suffix == ".onnx":
            self.backend = "onnx"
            self._load_onnx(self.policy_path)
        else:
            self.backend = "lerobot"
            self._load_lerobot(self.policy_path)

    def process(self, image: np.ndarray, state: Optional[np.ndarray] = None) -> Tuple[float, float]:
        if not self._action_queue:
            image_tensor = self._image_tensor(image)
            state_tensor = self._state_tensor(state)
            chunk = self._infer_action_chunk(image_tensor, state_tensor)
            for action in chunk[: self.n_action_steps]:
                self._action_queue.append(action)

        action_np = self._action_queue.popleft().reshape(-1)
        if action_np.shape[0] < 2:
            raise RuntimeError(f"ACT policy returned action with shape {action_np.shape}; expected at least 2 values")

        accel = float(np.clip(action_np[0], -1.0, 1.0))
        steer = float(np.clip(action_np[1], -1.0, 1.0))
        if self.control_mode != "ai":
            accel = self.acceleration
        return accel, steer

    def _load_onnx(self, path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX ACT inference") from exc

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.logger.info("Loaded ONNX ACT policy: %s providers=%s", path, self.session.get_providers())

    def _load_lerobot(self, path: Path) -> None:
        import torch

        try:
            from lerobot.policies.act import ACTPolicy
        except ImportError:
            from lerobot.policies.act.modeling_act import ACTPolicy

        self.torch = torch
        self.torch_device = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        self.policy = ACTPolicy.from_pretrained(path)
        self.policy.to(self.torch_device)
        self.policy.eval()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.logger.info("Loaded LeRobot ACT policy directory: %s", path)

    def _infer_action_chunk(self, image_tensor: np.ndarray, state_tensor: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            inputs = {"image": image_tensor}
            if "state" in self.input_names:
                inputs["state"] = state_tensor
            outputs = self.session.run(None, inputs)[0]
            return np.asarray(outputs[0], dtype=np.float32)

        torch = self.torch
        with torch.no_grad():
            batch = {
                "observation.images.front": torch.from_numpy(image_tensor).to(self.torch_device),
                "observation.state": torch.from_numpy(state_tensor).to(self.torch_device),
            }
            actions = self.policy.predict_action_chunk(batch)
        return actions.detach().cpu().numpy()[0].astype(np.float32)

    def _image_tensor(self, image: np.ndarray) -> np.ndarray:
        image = self._preprocess_image(image)
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def _state_tensor(self, state: Optional[np.ndarray]) -> np.ndarray:
        if state is None:
            state = np.zeros((2,), dtype=np.float32)
        return np.expand_dims(state.astype(np.float32), axis=0)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if self.crop_top_ratio > 0 or self.crop_bottom_ratio > 0:
            h = image.shape[0]
            top = int(h * self.crop_top_ratio)
            bottom = h - int(h * self.crop_bottom_ratio)
            image = image[top:bottom, :, :]
        return cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
