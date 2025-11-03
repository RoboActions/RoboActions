"""Remote policy client for retrieving RoboActions policy metadata."""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for sampling helpers
    torch = None  # type: ignore[assignment]

import requests

from ._version import __version__
from .config import ClientConfig, DEFAULT_TIMEOUT, RetryConfig
from .http import HttpClient
from .serialize import packb, unpackb


class PolicyStatus(Enum):
    """Enum representing health states returned by `/v1/healthz`."""

    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    DEPLOYING = "DEPLOYING"
    STOPPED = "STOPPED"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PolicyStatus":
        value = payload.get("status")
        if isinstance(value, str):
            normalized = value.strip().upper()
            member = cls.__members__.get(normalized)
            if member is not None:
                return member
        return cls.UNKNOWN


DEFAULT_REMOTE_POLICY_BASE_URL = "https://api.roboactions.com"

_BASE_POLICY_HEADERS: MutableMapping[str, str] = {
    "User-Agent": f"roboactions-sdk-python/{__version__}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


class RemotePolicy:
    """Fetch remote policy metadata from the RoboActions API."""

    def __init__(
        self,
        *,
        policy_id: str,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_REMOTE_POLICY_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        retries: Optional[RetryConfig] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        session: Optional[requests.Session] = None,
        http_client: Optional[HttpClient] = None,
    ) -> None:
        if not policy_id:
            raise ValueError("policy_id is required")

        # Read from environment variable if not explicitly provided
        if api_key is None:
            api_key = os.environ.get("ROBOACTIONS_API_KEY")

        if not api_key:
            raise ValueError(
                "api_key is required. Provide it explicitly or set the ROBOACTIONS_API_KEY environment variable."
            )

        self.policy_id = policy_id

        headers: MutableMapping[str, str] = dict(_BASE_POLICY_HEADERS)
        headers["x-policy-id"] = policy_id
        if default_headers:
            headers.update(default_headers)

        if http_client is not None:
            self._http = http_client
        else:
            config = ClientConfig(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=headers,
                retries=retries or RetryConfig(),
            )
            self._http = HttpClient(config=config, session=session)
        self._observation_shapes: List[Tuple[str, Tuple[int, ...]]] = []

    # ------------------------------------------------------------------
    # Context manager helpers

    def __enter__(self) -> "RemotePolicy":  # pragma: no cover - simple delegation
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP session if available."""

        close = getattr(self._http, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    # Public API

    def status(self, *, timeout: Optional[float] = None) -> PolicyStatus:
        """Return the policy's health status by calling `/v1/healthz`."""

        payload = self._get_json("/v1/healthz", timeout=timeout)
        return PolicyStatus.from_payload(payload)

    def ping(self) -> bool:
        """Simple health check to validate connectivity and credentials.

        Returns:
            True if the policy endpoint is reachable and credentials are valid,
            False otherwise.
        """

        try:
            self.status(timeout=None)
        except Exception:
            return False
        return True

    def wait_until_deployed(self, *, interval: float = 5.0) -> PolicyStatus:
        """Poll `status` until the policy leaves the `DEPLOYING` state."""

        if interval <= 0:
            raise ValueError("interval must be positive")

        while True:
            current_status = self.status(timeout=None)
            if current_status is not PolicyStatus.DEPLOYING:
                return current_status
            time.sleep(interval)

    def input_features(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Fetch the policy's expected input features."""

        payload = self._get_json("/v1/policy/input_features", timeout=timeout)
        self._observation_shapes = self._extract_observation_shapes(payload)
        return payload

    def output_features(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Fetch the policy's produced output features."""

        return self._get_json("/v1/policy/output_features", timeout=timeout)

    def sample_observation(self) -> Dict[str, "torch.Tensor"]:
        """Generate random observation tensors shaped for policy inference."""

        if torch is None:  # pragma: no cover - depends on optional torch install
            raise RuntimeError(
                "PyTorch is required for sample_observation; install `torch` to continue."
            )

        shapes = self.observation_shapes
        if not shapes:
            self.input_features()
            shapes = self.observation_shapes

        samples: Dict[str, "torch.Tensor"] = {}
        for key, shape in shapes:
            samples[key] = torch.rand(shape, dtype=torch.float32)
        samples["task"] = "Random Task"
        return samples

    def select_action(self, observation: Mapping[str, Any], *, timeout: Optional[float] = None) -> Any:
        """Send an observation batch to `/v1/policy/select_action` and return msgpack-decoded output."""

        return self._post_msgpack("/v1/policy/select_action", observation, timeout=timeout)

    def predict_action_chunk(
        self, observation: Mapping[str, Any], *, timeout: Optional[float] = None
    ) -> Any:
        """Send an observation batch to `/v1/policy/predict_action_chunk` and return decoded output."""

        return self._post_msgpack("/v1/policy/predict_action_chunk", observation, timeout=timeout)

    def reset(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Reset the policy and report result.

        On success returns `{ "reset": True }`.
        On failure returns `{ "reset": False, "error": "..." }`.
        """

        return self._get_json("/v1/policy/reset", timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers

    def _get_json(self, path: str, *, timeout: Optional[float]) -> Dict[str, Any]:
        payload = self._http.request("GET", path, timeout=timeout)
        # Ensure we always return a plain dict to callers.
        return dict(payload)

    def _post_msgpack(
        self, path: str, payload: Mapping[str, Any], *, timeout: Optional[float]
    ) -> Any:
        body = packb(payload)
        headers: MutableMapping[str, str] = {
            "Content-Type": "application/msgpack",
            "Accept": "application/msgpack",
        }

        config = getattr(self._http, "_config", None)
        if config is not None:
            auth = config.headers_with_auth().get("Authorization")
            if auth:
                headers["Authorization"] = auth

        response = self._http.request(
            "POST",
            path,
            data=body,
            headers=headers,
            timeout=timeout,
            expect_json=False,
        )
        if not isinstance(response, (bytes, bytearray)):
            raise TypeError("Expected binary response for msgpack request")
        return unpackb(response)

    @property
    def observation_shapes(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Return cached observation keys and shapes discovered in `input_features`."""

        return list(self._observation_shapes)

    @staticmethod
    def _extract_observation_shapes(payload: Mapping[str, Any]) -> List[Tuple[str, Tuple[int, ...]]]:
        """Collect observation keys and their numeric shapes from the payload."""

        def iter_items(source: Mapping[str, Any]) -> Iterable[Tuple[str, Any]]:
            features = source.get("input_features")
            if isinstance(features, Mapping):
                return features.items()
            return source.items()

        shapes: List[Tuple[str, Tuple[int, ...]]] = []
        for key, value in iter_items(payload):
            if not isinstance(key, str):
                continue
            if not key.startswith("observation"):
                continue
            if not isinstance(value, Mapping):
                continue
            raw_shape = value.get("shape")
            if not isinstance(raw_shape, (list, tuple)):
                continue
            normalized_shape: List[int] = []
            valid = True
            for dimension in raw_shape:
                if isinstance(dimension, int):
                    normalized_shape.append(dimension)
                else:
                    valid = False
                    break
            if valid and normalized_shape:
                shapes.append((key, tuple(normalized_shape)))
        return shapes
