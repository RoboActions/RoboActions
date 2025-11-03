"""Run a robosuite Lift task driven by a RemotePolicy and record a short video.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable

import imageio
import numpy as np
import robosuite as suite
import torch

from roboactions import RemotePolicy

# --- Configuration defaults -------------------------------------------------

# RemotePolicy deployment identifier.
# Please deploy https://huggingface.co/lerobot/pi05_libero_finetuned on the RoboActions platform
DEFAULT_POLICY_ID = "RWYh3T1VTTKqPoD-dGnAPw"

# robosuite camera names referenced throughout the tutorial.
FRONTVIEW_CAMERA = "frontview"
AGENTVIEW_CAMERA = "agentview"
EYE_IN_HAND_CAMERA = "robot0_eye_in_hand"

# Simulation / rendering defaults. These can be overridden from the CLI.
DEFAULT_OUTPUT = "video.mp4"
DEFAULT_FPS = 50
DEFAULT_CONTROL_FREQ = 20
DEFAULT_CAMERA_RESOLUTION = 256
DEFAULT_CHUNKS = 15
DEFAULT_ACTIONS_PER_CHUNK = 10


@dataclass
class SimulationConfig:
    """Collect configuration knobs so tutorials can present them in one place."""

    policy_id: str = DEFAULT_POLICY_ID
    output_path: str = DEFAULT_OUTPUT
    fps: int = DEFAULT_FPS
    control_freq: int = DEFAULT_CONTROL_FREQ
    camera_resolution: int = DEFAULT_CAMERA_RESOLUTION
    num_chunks: int = DEFAULT_CHUNKS
    actions_per_chunk: int = DEFAULT_ACTIONS_PER_CHUNK


def _quat2axisangle(quat: Iterable[float]) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to axis-angle representation.
    Taken from https://github.com/ARISE-Initiative/robosuite/blob/cb25aae7cac84c10409b96827a6c8d5a21f48f3a/robosuite/utils/transform_utils.py#L490
    """

    q = np.asarray(quat, dtype=np.float64).copy()
    # Guard against slight numerical drift that can push w outside [-1, 1].
    q[3] = np.clip(q[3], -1.0, 1.0)

    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        # Near-zero rotation: the axis is effectively undefined, return zeros.
        return np.zeros(3, dtype=np.float64)

    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _build_state(robosuite_obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten relevant robot state from a robosuite observation dict.
    Taken from https://github.com/Physical-Intelligence/openpi/blob/175f89c31d1b2631a8ff3b678768f17489c5ead4/examples/libero/main.py#L133
    """
    return np.concatenate(
        (
            robosuite_obs["robot0_eef_pos"],
            _quat2axisangle(robosuite_obs["robot0_eef_quat"]),
            robosuite_obs["robot0_gripper_qpos"],
        )
    )


def _to_chw_tensor(img_like) -> torch.Tensor:
    """Normalize image-like data into a float32 torch tensor with CxHxW layout."""
    arr = img_like if isinstance(img_like, np.ndarray) else np.asarray(img_like)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.from_numpy(arr.copy()) if isinstance(img_like, np.ndarray) else torch.as_tensor(arr)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor.float()
    return tensor


def _build_observation(robosuite_obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Match the RemotePolicy observation contract used in the tutorial."""
    state = torch.from_numpy(_build_state(robosuite_obs))
    agentview = _to_chw_tensor(robosuite_obs[f"{AGENTVIEW_CAMERA}_image"][::-1, ::-1])
    eye_in_hand = _to_chw_tensor(robosuite_obs[f"{EYE_IN_HAND_CAMERA}_image"][::-1, ::-1])
    task = "Pick up the red cube"
    return {
        "observation.state": torch.tensor(state, dtype=torch.float32),
        "observation.images.image": agentview,
        "observation.images.image2": eye_in_hand,
        "task": task,
    }


def _make_environment(cfg: SimulationConfig):
    """Create the robosuite Lift environment with tutorial-friendly defaults."""
    return suite.make(
        env_name="Lift",
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=[FRONTVIEW_CAMERA, AGENTVIEW_CAMERA, EYE_IN_HAND_CAMERA],
        control_freq=cfg.control_freq,
        camera_heights=cfg.camera_resolution,
        camera_widths=cfg.camera_resolution,
    )


def run_simulation(cfg: SimulationConfig) -> int:
    """Execute the policy in robosuite and record frames to ``cfg.output_path``.

    Returns
    -------
    int
        Number of frames written to the output video.
    """
    # Make sure to use an API key from the console and set ROBOACTIONS_API_KEY env variable.
    policy = RemotePolicy(policy_id=cfg.policy_id)
    policy.wait_until_deployed()

    env = _make_environment(cfg)
    frame_count = 0

    try:
        obs = env.reset()
        with imageio.get_writer(cfg.output_path, fps=cfg.fps, format="ffmpeg", codec="libx264") as writer:
            for chunk_idx in range(cfg.num_chunks):
                policy_obs = _build_observation(obs)
                action_chunk = policy.predict_action_chunk(policy_obs)
                predicted_actions = action_chunk[0]
                steps_this_chunk = min(cfg.actions_per_chunk, len(predicted_actions))

                for step_idx in range(steps_this_chunk):
                    action = predicted_actions[step_idx].cpu().numpy()
                    obs, _, _, _ = env.step(action)
                    frame = obs[f"{FRONTVIEW_CAMERA}_image"][::-1, ::-1]
                    writer.append_data(frame)
                    print(f"[chunk {chunk_idx + 1}/{cfg.num_chunks}] Saved frame #{frame_count}")
                    frame_count += 1
    finally:
        env.close()

    return frame_count


def parse_args() -> SimulationConfig:
    """Parse CLI flags so tutorials can demonstrate optional overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", default=DEFAULT_POLICY_ID, help="RemotePolicy deployment identifier.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output video filename (mp4).")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Video frames per second.")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_CHUNKS,
        help="Number of RemotePolicy action chunks to execute.",
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=DEFAULT_ACTIONS_PER_CHUNK,
        help="Maximum actions to take from each predicted chunk.",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=DEFAULT_CONTROL_FREQ,
        help="Control frequency passed to robosuite's environment factory.",
    )
    parser.add_argument(
        "--camera-resolution",
        type=int,
        default=DEFAULT_CAMERA_RESOLUTION,
        help="Square resolution (pixels) for offscreen cameras.",
    )
    args = parser.parse_args()
    return SimulationConfig(
        policy_id=args.policy_id,
        output_path=args.output,
        fps=args.fps,
        control_freq=args.control_freq,
        camera_resolution=args.camera_resolution,
        num_chunks=args.num_chunks,
        actions_per_chunk=args.actions_per_chunk,
    )


def main() -> None:
    """CLI entry point used throughout the tutorial."""
    config = parse_args()
    frame_count = run_simulation(config)
    print(f"Finished! Wrote {frame_count} frames to {config.output_path}")


if __name__ == "__main__":
    main()
