"""Gymnasium-like RemoteEnv over WebSocket for RoboActions `/remote_env`.

This client mirrors the common Gymnasium `Env` surface:
- reset(seed: Optional[int]) -> tuple[observation, info]
- step(action) -> tuple[observation, reward, terminated, truncated, info]
- render() -> np.ndarray | None, when created with render_mode="rgb_array"
- close() -> None
- action_space: gymnasium.spaces.Space

Transport:
- Connects to wss://.../remote_env with `Authorization: Bearer <API_KEY>`
- Frames are JSON-encoded text
- Rendered images returned as base64-encoded PNG; decoded to RGB ndarray
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - exercised in environments lacking gymnasium
    raise RuntimeError(
        "gymnasium is required for RemoteEnv. Install with `pip install gymnasium`."
    ) from exc

try:
    # websocket-client library
    import websocket  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - exercised in environments lacking websocket-client
    raise RuntimeError(
        "websocket-client is required for RemoteEnv. Install with `pip install websocket-client`."
    ) from exc

from ._version import __version__
from .config import DEFAULT_TIMEOUT
from .exceptions import (
    AuthenticationError,
    RoboActionsError,
    TransportError,
)


DEFAULT_REMOTE_ENV_BASE_URL = "https://api.roboactions.com"

# ---- Action space parsing ----------------------------------------------------

def _space_from_schema(schema: Mapping[str, Any]) -> "gym.spaces.Space":
    """Convert server action space schema into a gymnasium Space."""
    t = schema.get("type")
    if not isinstance(t, str):
        raise ValueError("Action space schema missing 'type'")

    t_upper = t.strip()
    if t_upper == "Discrete":
        n = schema.get("n")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("Discrete space requires positive integer 'n'")
        return gym.spaces.Discrete(n)

    if t_upper == "Box":
        shape = schema.get("shape")
        dtype_str = schema.get("dtype", "float32")
        low = schema.get("low")
        high = schema.get("high")
        if not isinstance(shape, (list, tuple)) or not all(isinstance(x, int) for x in shape):
            raise ValueError("Box space requires integer 'shape' list")
        try:
            dtype = np.dtype(dtype_str)
        except Exception as exc:
            raise ValueError(f"Invalid Box dtype: {dtype_str}") from exc
        if not isinstance(low, (list, tuple)) or not isinstance(high, (list, tuple)):
            raise ValueError("Box space requires 'low' and 'high' lists")
        low_arr = np.array(low, dtype=dtype)
        high_arr = np.array(high, dtype=dtype)
        if tuple(low_arr.shape) != tuple(shape) or tuple(high_arr.shape) != tuple(shape):
            raise ValueError("Box low/high must match 'shape'")
        return gym.spaces.Box(low=low_arr, high=high_arr, shape=tuple(shape), dtype=dtype)

    if t_upper == "MultiDiscrete":
        nvec = schema.get("nvec")
        if not isinstance(nvec, (list, tuple)) or not all(isinstance(x, int) and x > 0 for x in nvec):
            raise ValueError("MultiDiscrete space requires positive integer 'nvec' list")
        return gym.spaces.MultiDiscrete(nvec)

    if t_upper == "MultiBinary":
        n = schema.get("n")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("MultiBinary space requires positive integer 'n'")
        return gym.spaces.MultiBinary(n)

    if t_upper == "Dict":
        spaces = schema.get("spaces")
        if not isinstance(spaces, Mapping):
            raise ValueError("Dict space requires mapping 'spaces'")
        parsed: MutableMapping[str, "gym.spaces.Space"] = {}
        for key, child in spaces.items():
            if not isinstance(key, str):
                raise ValueError("Dict keys must be strings")
            if not isinstance(child, Mapping):
                raise ValueError("Dict 'spaces' values must be mappings")
            parsed[key] = _space_from_schema(child)
        return gym.spaces.Dict(parsed)

    if t_upper == "Tuple":
        spaces = schema.get("spaces")
        if not isinstance(spaces, (list, tuple)):
            raise ValueError("Tuple space requires list 'spaces'")
        parsed_list = []
        for child in spaces:
            if not isinstance(child, Mapping):
                raise ValueError("Tuple 'spaces' items must be mappings")
            parsed_list.append(_space_from_schema(child))
        return gym.spaces.Tuple(tuple(parsed_list))

    raise ValueError(f"Unsupported action space type: {t}")


def _decode_png_base64_to_ndarray(data_b64: str) -> np.ndarray:
    """Decode a base64 PNG string into an RGB numpy array HxWx3."""
    try:
        from PIL import Image  # lazy import to avoid import-time hard dependency
    except Exception as exc:  # pragma: no cover - exercised when Pillow is missing
        raise TransportError(
            "Pillow is required for render() decoding; install with `pip install Pillow`.",
            original=exc,
        )
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as exc:
        raise TransportError("Failed to base64-decode render() payload", original=exc)
    try:
        with Image.open(io.BytesIO(raw)) as img:
            rgb = img.convert("RGB")
            return np.asarray(rgb)
    except Exception as exc:
        raise TransportError("Failed to decode PNG render() payload", original=exc)


# ---- RemoteEnv ---------------------------------------------------------------

class RemoteEnv:
    """Remote Gymnasium environment over WebSocket.

    Example:
        env = RemoteEnv(\"CartPole-v1\", render_mode=\"rgb_array\", api_key=\"rk_...\")
        obs, info = env.reset(seed=123)
        obs, reward, terminated, truncated, info = env.step(0)
        frame = env.render()  # ndarray or None
        env.close()
    """

    def __init__(
        self,
        env_id: str,
        *,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_REMOTE_ENV_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        default_headers: Optional[Mapping[str, str]] = None,
        _ws_override: Optional[Any] = None,
    ) -> None:
        if not isinstance(env_id, str) or not env_id.strip():
            raise ValueError("env_id is required")
        self._env_id = env_id.strip()

        if api_key is None:
            api_key = os.environ.get("ROBOACTIONS_API_KEY")
        if not api_key:
            raise ValueError(
                "api_key is required. Provide it explicitly or set ROBOACTIONS_API_KEY."
            )
        self._api_key = api_key

        self._render_mode = render_mode
        self._timeout = float(timeout)
        self._base_url = base_url
        self._ws: Optional[Any] = None
        self._needs_reset = False
        self._action_space: Optional["gym.spaces.Space"] = None

        # Connect and make the environment
        self._connect(_ws_override=_ws_override, default_headers=default_headers)
        self._send_json(
            {
                "op": "make",
                "env_id": self._env_id,
                **({"render_mode": self._render_mode} if self._render_mode else {}),
                **({"seed": int(seed)} if seed is not None else {}),
            }
        )
        payload = self._recv_json()
        if payload.get("type") == "error":
            self._raise_error(payload)
        if payload.get("type") != "make_ok":
            raise TransportError(f"Unexpected response to make: {payload}")

        action_schema = payload.get("action_space")
        if not isinstance(action_schema, Mapping):
            raise TransportError("make_ok missing 'action_space'")
        self._action_space = _space_from_schema(action_schema)

    # ---------------- Context manager ----------------
    def __enter__(self) -> "RemoteEnv":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()

    # ---------------- Properties ----------------
    @property
    def action_space(self) -> "gym.spaces.Space":
        if self._action_space is None:
            raise RuntimeError("Environment not initialized with action space")
        return self._action_space

    # ---------------- Public API ----------------
    def reset(self, *, seed: Optional[int] = None) -> Tuple[Any, Mapping[str, Any]]:
        """Reset the remote environment."""
        self._ensure_open()
        self._send_json({"op": "reset", **({"seed": int(seed)} if seed is not None else {})})
        payload = self._recv_json()
        if payload.get("type") == "error":
            self._raise_error(payload)
        if payload.get("type") != "reset_ok":
            raise TransportError(f"Unexpected response to reset: {payload}")
        self._needs_reset = False
        observation = payload.get("observation")
        info = payload.get("info") or {}
        if not isinstance(info, Mapping):
            info = {}
        return observation, dict(info)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Mapping[str, Any]]:
        """Step the environment one tick using a valid action."""
        self._ensure_open()
        if self._needs_reset:
            raise RoboActionsError("NEEDS_RESET: Call reset() before step()")
        # Pre-validate quickly on client side
        if self._action_space is not None and not self._action_space.contains(action):
            raise RoboActionsError("INVALID_ACTION: Action not contained in action_space")

        self._send_json({"op": "step", "action": action})
        payload = self._recv_json()
        if payload.get("type") == "error":
            self._raise_error(payload)
        if payload.get("type") != "step_ok":
            raise TransportError(f"Unexpected response to step: {payload}")

        observation = payload.get("observation")
        reward = float(payload.get("reward", 0.0))
        terminated = bool(payload.get("terminated", False))
        truncated = bool(payload.get("truncated", False))
        info = payload.get("info") or {}
        if not isinstance(info, Mapping):
            info = {}

        if terminated or truncated:
            self._needs_reset = True

        return observation, reward, terminated, truncated, dict(info)

    def render(self) -> Optional[np.ndarray]:
        """Render a frame as a numpy RGB array when render_mode == 'rgb_array'."""
        self._ensure_open()
        if self._render_mode != "rgb_array":
            return None
        self._send_json({"op": "render", "encoding": "png_base64"})
        payload = self._recv_json()
        if payload.get("type") == "error":
            self._raise_error(payload)
        if payload.get("type") != "render":
            raise TransportError(f"Unexpected response to render: {payload}")
        data = payload.get("data")
        if not isinstance(data, str):
            raise TransportError("render response missing 'data'")
        return _decode_png_base64_to_ndarray(data)

    def close(self) -> None:
        """Close the environment and underlying WebSocket."""
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            try:
                self._send_json({"op": "close"}, ws_obj=ws)
                payload = self._recv_json(ws_obj=ws)
                if payload.get("type") == "error":
                    # Ignore server-side error during close; still close socket
                    pass
            except Exception:
                # Ignore protocol errors during shutdown
                pass
        finally:
            try:
                ws.close()
            except Exception:
                pass

    # ---------------- Internals ----------------
    def _connect(
        self,
        *,
        _ws_override: Optional[Any],
        default_headers: Optional[Mapping[str, str]],
    ) -> None:
        if _ws_override is not None:
            self._ws = _ws_override
            return
        url = self._build_ws_url(self._base_url)
        headers: MutableMapping[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"roboactions-sdk-python/{__version__}",
        }
        if default_headers:
            headers.update(default_headers)
        header_list = [f"{k}: {v}" for k, v in headers.items()]
        try:
            self._ws = websocket.create_connection(url, header=header_list, timeout=self._timeout)
        except Exception as exc:
            raise TransportError("Failed to open WebSocket connection", original=exc) from exc

    @staticmethod
    def _build_ws_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.startswith("http://"):
            return f"ws://{base[len('http://'):]}/remote_env"
        if base.startswith("https://"):
            return f"wss://{base[len('https://'):]}/remote_env"
        # Assume already ws(s)://
        if base.startswith("ws://") or base.startswith("wss://"):
            return f"{base}/remote_env"
        # Fallback
        return f"wss://{base}/remote_env"

    def _ensure_open(self) -> None:
        if self._ws is None:
            raise RoboActionsError("NOT_INITIALIZED: Environment is closed")

    @staticmethod
    def _to_json_safe(obj: Any) -> Any:
        """Convert numpy scalars/arrays and containers into JSON-serializable types."""
        # numpy scalar types
        if isinstance(obj, np.generic):
            return obj.item()
        # numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # containers
        if isinstance(obj, (list, tuple)):
            return [RemoteEnv._to_json_safe(x) for x in obj]
        if isinstance(obj, Mapping):
            return {str(k): RemoteEnv._to_json_safe(v) for k, v in obj.items()}
        return obj

    def _send_json(self, obj: Mapping[str, Any], *, ws_obj: Optional[Any] = None) -> None:
        ws = ws_obj or self._ws
        if ws is None:
            raise RoboActionsError("NOT_INITIALIZED: Environment is closed")
        try:
            ws.send(json.dumps(self._to_json_safe(obj)))
        except Exception as exc:
            raise TransportError("Failed to send WebSocket frame", original=exc) from exc

    def _recv_json(self, *, ws_obj: Optional[Any] = None) -> Mapping[str, Any]:
        ws = ws_obj or self._ws
        if ws is None:
            raise RoboActionsError("NOT_INITIALIZED: Environment is closed")
        try:
            frame = ws.recv()
        except Exception as exc:
            # Best-effort authentication mapping if server closes with 1008
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if code == 1008:
                raise AuthenticationError("Authentication failed (WebSocket 1008)")
            raise TransportError("Failed to receive WebSocket frame", original=exc) from exc
        try:
            data = json.loads(frame)
        except Exception as exc:
            raise TransportError("Failed to decode JSON frame", original=exc) from exc
        if not isinstance(data, Mapping):
            raise TransportError("Expected mapping JSON frame from server")
        return data

    @staticmethod
    def _raise_error(payload: Mapping[str, Any]) -> None:
        code = payload.get("code") or "ERROR"
        message = payload.get("message") or "RemoteEnv error"
        full_message = f"{code}: {message}"
        # Map specific server codes to clearer messages where helpful
        if code == "NEEDS_RESET":
            raise RoboActionsError("NEEDS_RESET: Call reset() before step()")
        if code == "INVALID_ACTION":
            raise RoboActionsError("INVALID_ACTION: Action not contained in action_space")
        if code == "NOT_INITIALIZED":
            raise RoboActionsError("NOT_INITIALIZED: Call make/reset first")
        if code == "RENDER_UNAVAILABLE":
            raise RoboActionsError("RENDER_UNAVAILABLE: Create with render_mode='rgb_array'")
        raise RoboActionsError(full_message, payload=dict(payload))


