#!/usr/bin/env python3
"""Export a trained LeRobot ACT policy to ONNX for ROS runtime inference.

The exported graph includes the dataset normalizer and action unnormalizer, so
the ROS node only needs to feed an RGB image tensor in [0, 1] and raw vehicle
state values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

try:
    from lerobot.policies.act import ACTPolicy
except ImportError:
    from lerobot.policies.act.modeling_act import ACTPolicy

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class ActOnnxWrapper(torch.nn.Module):
    def __init__(self, policy: ACTPolicy, stats_path: Path):
        super().__init__()
        self.model = policy.model
        stats = load_file(str(stats_path))

        self.register_buffer("image_mean", stats["observation.images.front.mean"].view(1, 3, 1, 1))
        self.register_buffer("image_std", stats["observation.images.front.std"].view(1, 3, 1, 1))
        self.register_buffer("state_mean", stats["observation.state.mean"].view(1, -1))
        self.register_buffer("state_std", stats["observation.state.std"].view(1, -1))
        self.register_buffer("action_mean", stats["action.mean"].view(1, 1, -1))
        self.register_buffer("action_std", stats["action.std"].view(1, 1, -1))

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image = (image - self.image_mean) / self.image_std.clamp_min(1e-6)
        state = (state - self.state_mean) / self.state_std.clamp_min(1e-6)
        actions, _ = self.model({OBS_IMAGES: [image], OBS_STATE: state})
        return actions * self.action_std + self.action_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./ckpt/act_policy.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.eval()
    policy.model.eval()

    stats_path = args.policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    wrapper = ActOnnxWrapper(policy, stats_path).eval()
    try:
        device = next(wrapper.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    wrapper.to(device)

    image_feature = policy.config.input_features["observation.images.front"]
    state_feature = policy.config.input_features["observation.state"]
    _, height, width = image_feature.shape
    state_dim = state_feature.shape[0]

    dummy_image = torch.zeros(1, 3, height, width, dtype=torch.float32, device=device)
    dummy_state = torch.zeros(1, state_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_state),
            args.output,
            input_names=["image", "state"],
            output_names=["action_chunk"],
            dynamic_axes={
                "image": {0: "batch"},
                "state": {0: "batch"},
                "action_chunk": {0: "batch"},
            },
            opset_version=args.opset,
            dynamo=False,
        )

    print(f"Exported ONNX policy: {args.output}")


if __name__ == "__main__":
    main()
