#!/usr/bin/env python3
"""HTTP inference server for LeRobot SmolVLA policies.

Run this from the host conda environment. The ROS2 bridge node runs in the
AI Challenge Docker container and calls this server over localhost.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.policies import get_policy_class
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


LOGGER = logging.getLogger("smolvla_inference_server")


class SmolVLAInference:
    def __init__(self, policy_path: Path, device: str, task: str):
        self.policy_path = policy_path
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.task = task

        self.policy = get_policy_class("smolvla").from_pretrained(policy_path)
        self.policy.to(self.device)
        self.policy.eval()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            policy_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            policy_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        )

        LOGGER.info("Loaded SmolVLA policy: %s device=%s", policy_path, self.device)

    def predict(self, image_b64: str, state: list[float] | None, task: str | None) -> dict[str, Any]:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        if state is None:
            state_tensor = torch.zeros((2,), dtype=torch.float32)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)

        raw_batch: dict[str, Any] = {
            "observation.images.front": image_tensor,
            "observation.state": state_tensor,
            "task": task or self.task,
        }

        with torch.no_grad():
            batch = self.preprocessor(raw_batch)
            actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(actions)

        action_chunk = actions.detach().cpu().numpy().astype(np.float32)
        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]
        elif action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]

        return {
            "action": action_chunk[0].tolist(),
            "action_chunk": action_chunk.tolist(),
        }


class Handler(BaseHTTPRequestHandler):
    engine: SmolVLAInference

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            result = self.engine.predict(
                image_b64=payload["image_jpeg_b64"],
                state=payload.get("state"),
                task=payload.get("task"),
            )
            self._write_json(result)
        except Exception as exc:
            LOGGER.exception("prediction failed")
            self._write_json({"error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug(fmt, *args)

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="drive the racing kart around the course")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    Handler.engine = SmolVLAInference(args.policy_path, args.device, args.task)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("Serving SmolVLA HTTP inference at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
