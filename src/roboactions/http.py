"""Internal HTTP client with retry + error handling."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, MutableMapping, Optional, Union

import requests

from .config import ClientConfig
from .exceptions import (
    HTTP_STATUS_TO_ERROR,
    ClientError,
    RateLimitError,
    RoboActionsError,
    ServerError,
    TransportError,
)

JsonMapping = Mapping[str, Any]


class HttpClient:
    """Thin wrapper around `requests.Session` with RoboActions defaults."""

    def __init__(self, config: ClientConfig, session: Optional[requests.Session] = None) -> None:
        self._config = config
        self._session = session or requests.Session()

    def close(self) -> None:
        self._session.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[JsonMapping] = None,
        json_body: Optional[JsonMapping] = None,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        expect_json: bool = True,
    ) -> Union[JsonMapping, bytes]:
        if json_body is not None and data is not None:
            raise ValueError("Cannot supply both json_body and data")

        url = self._build_url(path)
        merged_headers = self._merge_headers(headers)
        attempt = 0

        while True:
            try:
                response = self._session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=None if data is not None else json_body,
                    data=data,
                    headers=merged_headers,
                    timeout=timeout or self._config.timeout,
                )
            except requests.Timeout as exc:
                raise TransportError("Request timed out", original=exc) from exc
            except requests.RequestException as exc:
                raise TransportError("Networking error", original=exc) from exc

            if 200 <= response.status_code < 300:
                if expect_json:
                    return self._parse_json(response)
                return response.content

            if self._should_retry(response.status_code, attempt):
                attempt += 1
                self._sleep(attempt)
                continue

            self._raise_for_status(response)

    # ----- internal helpers -------------------------------------------------

    def _build_url(self, path: str) -> str:
        base = self._config.base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"

    def _merge_headers(self, headers: Optional[Mapping[str, str]]) -> MutableMapping[str, str]:
        merged: MutableMapping[str, str] = dict(self._config.headers_with_auth())
        if headers:
            merged.update(headers)
        return merged

    def _parse_json(self, response: requests.Response) -> JsonMapping:
        if not response.content:
            return {}
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise TransportError("Failed to decode JSON response", original=exc) from exc

    def _should_retry(self, status_code: int, attempt: int) -> bool:
        return self._config.retries.should_retry(status_code, attempt)

    def _sleep(self, attempt: int) -> None:
        delay = self._config.retries.backoff_factor * (2 ** (attempt - 1))
        time.sleep(delay)

    def _raise_for_status(self, response: requests.Response) -> None:
        status = response.status_code
        payload = self._safe_error_payload(response)

        if status >= 500:
            raise ServerError("RoboActions server error", payload=payload)

        exc_type = HTTP_STATUS_TO_ERROR.get(status, ClientError)

        if exc_type is RateLimitError:
            retry_after = float(response.headers.get("Retry-After", "0")) or None
            raise RateLimitError("Rate limit exceeded", retry_after=retry_after)

        if issubclass(exc_type, RoboActionsError):
            message = self._primary_error_message(payload)
            if not message:
                message = "RoboActions API error"
            raise exc_type(message, payload=payload)

        raise ClientError("Unexpected HTTP error", payload=payload)

    @staticmethod
    def _safe_error_payload(response: requests.Response) -> JsonMapping:
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"message": response.text or "Unknown error"}

    @staticmethod
    def _primary_error_message(payload: Mapping[str, Any]) -> Optional[str]:
        """Extract the most helpful error message from the API payload."""

        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()

        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, Mapping):
                    msg = item.get("message") or item.get("msg")
                    if isinstance(msg, str) and msg.strip():
                        parts.append(msg.strip())
            if parts:
                return "; ".join(parts)

        return None
