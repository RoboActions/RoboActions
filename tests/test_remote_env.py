import base64
import json

import numpy as np
import pytest

from roboactions.remote_env import RemoteEnv


class StubWS:
    def __init__(self, frames=None, raise_on_recv=None):
        self.frames = list(frames or [])
        self.raise_on_recv = raise_on_recv
        self.sent = []
        self.closed = False

    def send(self, data):
        try:
            self.sent.append(json.loads(data))
        except Exception:
            self.sent.append(data)

    def recv(self):
        if self.raise_on_recv:
            raise self.raise_on_recv
        if not self.frames:
            raise AssertionError("No stubbed frame to recv")
        return self.frames.pop(0)

    def close(self):
        self.closed = True


def make_ok_frame_discrete2():
    return json.dumps(
        {
            "type": "make_ok",
            "action_space": {"type": "Discrete", "n": 2},
            "observation": [0.0],
            "info": {},
        }
    )


def reset_ok_frame(obs=None):
    return json.dumps({"type": "reset_ok", "observation": obs if obs is not None else [1.0], "info": {}})


def step_ok_frame(obs=None, reward=1.0, terminated=False, truncated=False):
    return json.dumps(
        {
            "type": "step_ok",
            "observation": obs if obs is not None else [2.0],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": {},
        }
    )


def render_frame_png_b64_one_px():
    # 1x1 PNG (opaque black) base64
    data_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )
    return json.dumps({"type": "render", "format": "png_base64", "data": data_b64})


def auth_error_exc():
    class AuthClosed(Exception):
        status_code = 1008

    return AuthClosed("closed")


def test_make_reset_step_flow(monkeypatch):
    frames = [
        make_ok_frame_discrete2(),
        reset_ok_frame([3.14]),
        step_ok_frame([2.72], reward=0.5, terminated=False, truncated=False),
    ]
    stub = StubWS(frames=frames)

    def fake_create_connection(url, header=None, timeout=None):
        assert url.endswith("/remote_env")
        # Check auth header passed through
        assert any(h.startswith("Authorization: Bearer rk_test") for h in (header or []))
        return stub

    monkeypatch.setattr("roboactions.remote_env.websocket.create_connection", fake_create_connection)

    env = RemoteEnv("CartPole-v1", api_key="rk_test")
    # action_space parsed
    assert env.action_space.n == 2

    obs, info = env.reset()
    assert obs == [3.14]
    assert info == {}

    obs, reward, terminated, truncated, info = env.step(1)
    assert obs == [2.72]
    assert reward == 0.5
    assert not terminated
    assert not truncated
    assert info == {}

    # Ensure ops were sent in order: make, reset, step
    assert [m["op"] for m in stub.sent] == ["make", "reset", "step"]


def test_needs_reset_enforced(monkeypatch):
    frames = [
        make_ok_frame_discrete2(),
        step_ok_frame([0.0], reward=1.0, terminated=True, truncated=False),
    ]
    stub = StubWS(frames=frames)
    monkeypatch.setattr(
        "roboactions.remote_env.websocket.create_connection", lambda url, header=None, timeout=None: stub
    )

    env = RemoteEnv("CartPole-v1", api_key="rk_test")
    # First step ends episode
    env.step(0)
    with pytest.raises(Exception, match="NEEDS_RESET"):
        env.step(0)


def test_invalid_action_prevalidation(monkeypatch):
    frames = [make_ok_frame_discrete2()]
    stub = StubWS(frames=frames)
    monkeypatch.setattr(
        "roboactions.remote_env.websocket.create_connection", lambda url, header=None, timeout=None: stub
    )

    env = RemoteEnv("CartPole-v1", api_key="rk_test")
    with pytest.raises(Exception, match="INVALID_ACTION"):
        env.step(3)  # invalid for Discrete(2)
    # Ensure no 'step' was sent to wire
    assert [m["op"] for m in stub.sent] == ["make"]


def test_render_rgb_array(monkeypatch):
    # Skip this test if Pillow is not available
    try:
        import PIL  # noqa: F401
    except Exception:
        pytest.skip("Pillow not installed; skipping render test")
    frames = [make_ok_frame_discrete2(), render_frame_png_b64_one_px()]
    stub = StubWS(frames=frames)
    monkeypatch.setattr(
        "roboactions.remote_env.websocket.create_connection", lambda url, header=None, timeout=None: stub
    )

    env = RemoteEnv("CartPole-v1", api_key="rk_test", render_mode="rgb_array")
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
    assert [m["op"] for m in stub.sent] == ["make", "render"]


def test_render_none_when_not_enabled(monkeypatch):
    frames = [make_ok_frame_discrete2()]
    stub = StubWS(frames=frames)
    monkeypatch.setattr(
        "roboactions.remote_env.websocket.create_connection", lambda url, header=None, timeout=None: stub
    )

    env = RemoteEnv("CartPole-v1", api_key="rk_test")
    assert env.render() is None
    # No 'render' op sent
    assert [m["op"] for m in stub.sent] == ["make"]


def test_close_sends_close(monkeypatch):
    frames = [make_ok_frame_discrete2(), json.dumps({"type": "close_ok"})]
    stub = StubWS(frames=frames)
    monkeypatch.setattr(
        "roboactions.remote_env.websocket.create_connection", lambda url, header=None, timeout=None: stub
    )

    env = RemoteEnv("CartPole-v1", api_key="rk_test")
    env.close()
    assert stub.closed is True
    assert [m["op"] for m in stub.sent] == ["make", "close"]


def test_authentication_close_1008(monkeypatch):
    stub = StubWS(frames=[], raise_on_recv=auth_error_exc())

    def fake_create_connection(url, header=None, timeout=None):
        return stub

    monkeypatch.setattr("roboactions.remote_env.websocket.create_connection", fake_create_connection)

    with pytest.raises(Exception) as ei:
        RemoteEnv("CartPole-v1", api_key="rk_test")
    assert "Authentication failed" in str(ei.value)


