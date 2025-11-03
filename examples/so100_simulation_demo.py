"""Evaluate a remote policy on SO100 tasks and capture a tutorial-friendly video.

This script connects to a remote policy hosted via ``roboactions.RemotePolicy`` and
rolls it out inside the SO100 Gymnasium environment. The full trajectory is saved
as an ``.mp4`` so it can be embedded in walkthroughs or tutorials.

Typical usage (after installing the package and dependencies)::

    python scripts/evaluate_remote_policy.py --policy-id <REMOTE_ID>

Use ``--help`` to see all configurable options.

From-scratch setup with ``uv``::

    git clone https://github.com/your-org/SO100-Simulation.git
    cd SO100-Simulation
    # install uv: https://astral.sh/uv (macOS/Linux quick install below)
    curl -Ls https://astral.sh/uv/install.sh | sh
    uv python install 3.10
    uv venv --python 3.10 .venv && source .venv/bin/activate
    uv pip install -e .
    python scripts/example.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import gymnasium as gym
import imageio
import numpy as np
import torch

import gym_so100  # noqa: F401  (ensures SO100 environments are registered)
from gym_so100.constants import (
    normalize_gym_so100_to_lerobot,
    normalize_lerobot_to_gym_so100,
)
from roboactions import RemotePolicy

MAX_STEPS_PER_EPISODE = 150


def parse_args() -> argparse.Namespace:
    """Build and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Roll out a remote policy in SO100 and record the evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="dua0Zv6tTcS7kpspb-rpoQ",
        help="Identifier of the remote policy to evaluate",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gym_so100/SO100CubeToBin-v0",
        help="Registered Gymnasium task id to evaluate.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up the red cube and put it in the bin.",
        help="Natural-language prompt supplied to the remote policy.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply lerobot ↔ SO100 action normalization on observations and outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=41,
        help="Base seed for environment resets. Episodes use seed + episode index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/videos"),
        help="Directory where the evaluation video will be stored.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="remote_policy.mp4",
        help="Filename for the saved mp4 video.",
    )
    return parser.parse_args()


def to_policy_input(
    observation: Dict[str, np.ndarray],
    prompt: str,
    normalize: bool,
) -> Dict[str, torch.Tensor | str]:
    """Convert a Gymnasium observation into the structure expected by the policy."""
    agent_pos = observation["agent_pos"].copy()
    if normalize:
        agent_pos = normalize_gym_so100_to_lerobot(agent_pos)

    state = torch.from_numpy(agent_pos).to(torch.float32).unsqueeze(0)

    image = torch.from_numpy(observation["pixels"]).to(torch.float32)
    image = image.permute(2, 0, 1).unsqueeze(0) / 255.0

    return {
        "observation.state": state,
        "observation.images.camera1": image,
        "task": prompt,
    }


def convert_action(
    tensor_action: torch.Tensor,
    normalize: bool,
) -> Iterable[np.ndarray]:
    """Convert a sequence of torch action tensors into numpy arrays."""
    numpy_action = tensor_action.squeeze(0).detach().cpu().numpy()
    if normalize:
        numpy_action = normalize_lerobot_to_gym_so100(numpy_action.copy())
    return numpy_action


def evaluate_policy(
    policy: RemotePolicy,
    env: gym.Env,
    *,
    num_episodes: int,
    prompt: str,
    normalize: bool,
    base_seed: int,
) -> tuple[List[np.ndarray], List[float], float]:
    """Run evaluation episodes and collect rendered frames."""
    all_frames: List[np.ndarray] = []
    total_rewards: List[float] = []
    best_reward = float("-inf")

    for episode_idx in range(num_episodes):
        print(f"Starting episode {episode_idx + 1} / {num_episodes}")

        policy.reset()
        observation, _ = env.reset(seed=base_seed + episode_idx)
        episode_frames = [env.render()]
        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS_PER_EPISODE:
            policy_input = to_policy_input(observation, prompt, normalize)
            action_chunks = policy.predict_action_chunk(policy_input)
            for i in range(10):
                action = convert_action(action_chunks[0][i], False)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_frames.append(env.render())

                episode_reward += reward
                best_reward = max(best_reward, episode_reward)

                done = terminated or truncated
                steps += 1
                if done or steps >= MAX_STEPS_PER_EPISODE:
                    break
        outcome = "Success" if done else "Failure"
        print(f"Episode {episode_idx + 1} finished | reward={episode_reward:.2f} | {outcome}")

        all_frames.extend(episode_frames)
        total_rewards.append(episode_reward)

    return all_frames, total_rewards, best_reward


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to remote policy...")
    policy = RemotePolicy(policy_id=args.policy_id)
    policy.wait_until_deployed()
    print("Policy deployed.")

    torch.set_default_dtype(torch.float32)

    env = gym.make(
        args.task,
        obs_type="so100_pixels_agent_pos",
        observation_width=640,
        observation_height=480,
    )

    print("Policy input signature:", policy.input_features())
    print("Environment observation space:", env.observation_space)
    print("Policy output signature:", policy.output_features())
    print("Environment action space:", env.action_space)

    frames, rewards, best_reward = evaluate_policy(
        policy,
        env,
        num_episodes=args.num_episodes,
        prompt=args.prompt,
        normalize=args.normalize,
        base_seed=args.seed,
    )

    fps = env.metadata["render_fps"]
    video_path = args.output_dir / args.video_name
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

    print(f"Saved evaluation video to {video_path}")
    print(f"Best episode reward: {best_reward:.2f}")
    print(f"Average episode reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
