"""Lightweight Gym-like façade for creating RemoteEnv instances.

Usage:
    from roboactions import remotegym
    env = remotegym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset()
"""
from __future__ import annotations

from typing import Mapping, Optional

from .config import RetryConfig


def make(
    env_id: str,
    *,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    retries: Optional[RetryConfig] = None,
    default_headers: Optional[Mapping[str, str]] = None,
):
    """Create and return a remote Gymnasium environment.

    This mirrors the ergonomics of gym.make(...) but returns a network-backed RemoteEnv.
    Heavy/optional dependencies are imported lazily when called.
    """
    # Lazy import so importing 'remotegym' does not require optional deps
    from .remote_env import RemoteEnv, DEFAULT_REMOTE_ENV_BASE_URL  # type: ignore
    return RemoteEnv(
        env_id,
        render_mode=render_mode,
        seed=seed,
        api_key=api_key,
        base_url=base_url or DEFAULT_REMOTE_ENV_BASE_URL,
        timeout=timeout if timeout is not None else 10.0,  # default aligns with DEFAULT_TIMEOUT
        retries=retries,
        default_headers=default_headers,
    )


