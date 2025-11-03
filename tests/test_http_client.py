from __future__ import annotations

import json

import pytest

from roboactions.config import ClientConfig
from roboactions.exceptions import AuthenticationError
from roboactions.http import HttpClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.content = json.dumps(payload).encode()
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload


def test_authentication_error_surfaces_backend_detail() -> None:
    config = ClientConfig(api_key="rk_test")
    client = HttpClient(config)

    response = FakeResponse(401, {"detail": "Invalid API key."})

    with pytest.raises(AuthenticationError) as exc:
        client._raise_for_status(response)  # type: ignore[arg-type]

    assert str(exc.value) == "Invalid API key."
    assert exc.value.payload["detail"] == "Invalid API key."

