"""RoboActions Python SDK public interface."""

from ._version import __version__
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    RoboActionsError,
)
from .policy import PolicyStatus, PolicySummary, RemotePolicy
from . import remotegym as remotegym  # type: ignore[attr-defined]
try:
    # Lazy import so base users don't need optional WS/render deps
    from .remote_env import RemoteEnv  # type: ignore[attr-defined]
except Exception:
    class RemoteEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "RemoteEnv requires optional dependencies. Install with:\n"
                "  pip install gymnasium websocket-client Pillow\n"
                "and then retry."
            )
__all__ = [
    "RemoteEnv",
    "RemotePolicy",
    "PolicyStatus",
    "PolicySummary",
    "RoboActionsError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "RateLimitError",
    "__version__",
    "remotegym",
]
