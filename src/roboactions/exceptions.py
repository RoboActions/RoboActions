"""Custom exception hierarchy for RoboActions related failures."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class RoboActionsError(Exception):
    """Base exception for all SDK errors."""

    def __init__(self, message: str, *, payload: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.payload = dict(payload or {})


class AuthenticationError(RoboActionsError):
    """Raised when API key authentication fails."""


class AuthorizationError(RoboActionsError):
    """Raised when the client is not permitted to access a resource."""


class NotFoundError(RoboActionsError):
    """Raised when a requested resource does not exist."""


class RateLimitError(RoboActionsError):
    """Raised when the API indicates the client is being throttled."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None) -> None:
        super().__init__(message, payload={"retry_after": retry_after} if retry_after else None)
        self.retry_after = retry_after


class ServerError(RoboActionsError):
    """Raised for 5xx responses."""


class ClientError(RoboActionsError):
    """Raised for 4xx responses other than auth, not found, or rate limits."""


class TransportError(RoboActionsError):
    """Raised for networking, serialization, or timeout issues."""

    def __init__(self, message: str, *, original: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.original = original


HTTP_STATUS_TO_ERROR = {
    400: ClientError,
    401: AuthenticationError,
    403: AuthorizationError,
    404: NotFoundError,
    408: TransportError,
    409: ClientError,
    422: ClientError,
    429: RateLimitError,
}
