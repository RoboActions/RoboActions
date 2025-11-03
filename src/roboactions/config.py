"""Configuration primitives for the RoboActions SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

from ._version import __version__

DEFAULT_BASE_URL = "https://api.roboactions.com/v1"
DEFAULT_TIMEOUT = 60.0


@dataclass(slots=True)
class RetryConfig:
    """Basic retry strategy configuration for idempotent requests."""

    enabled: bool = True
    max_attempts: int = 3
    backoff_factor: float = 0.5
    status_forcelist: tuple[int, ...] = (408, 429, 500, 502, 503, 504)

    def should_retry(self, status_code: int, attempt: int) -> bool:
        return (
            self.enabled
            and attempt < self.max_attempts
            and status_code in self.status_forcelist
        )


@dataclass(slots=True)
class ClientConfig:
    """Runtime configuration for the HTTP client."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    default_headers: MutableMapping[str, str] = field(
        default_factory=lambda: {
            "User-Agent": f"roboactions-sdk-python/{__version__}",
            "Content-Type": "application/json",
        }
    )
    retries: RetryConfig = field(default_factory=RetryConfig)

    def headers_with_auth(self) -> Mapping[str, str]:
        headers = dict(self.default_headers)
        headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers
