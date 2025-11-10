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
            "observation_space": {
                "type": "Box",
                "shape": [1],
                "dtype": "float32",
                "low": [0.0],
                "high": [1.0],
            },
            "metadata": {"render_modes": ["rgb_array"], "render_fps": 30},
            "spec": {"id": "CartPole-v1", "max_episode_steps": 500},
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
    frames = [make_ok_frame_discrete2(), step_ok_frame([2.72], reward=0.5, terminated=False, truncated=False)]
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
    # observation_space parsed
    assert env.observation_space.shape == (1,)
    assert env.observation_space.dtype == np.float32
    # render_mode and reward_range per Gymnasium API
    assert env.render_mode is None
    assert env.reward_range[0] == -float("inf") and env.reward_range[1] == float("inf")
    # initial observation/info available immediately after make
    assert env.initial_observation == [0.0]
    assert env.initial_info == {}
    # spec should be None; raw spec available via server_spec
    assert env.spec is None
    assert env.server_spec.get("id") == "CartPole-v1"
    assert env.server_spec.get("max_episode_steps") == 500

    obs, reward, terminated, truncated, info = env.step(1)
    assert obs == [2.72]
    assert reward == 0.5
    assert not terminated
    assert not truncated
    assert info == {}

    # Ensure ops were sent in order: make, step
    assert [m["op"] for m in stub.sent] == ["make", "step"]


def test_reset_short_circuit_after_make(monkeypatch):
    frames = [make_ok_frame_discrete2()]
    stub = StubWS(frames=frames)

    def fake_create_connection(url, header=None, timeout=None):
        return stub

    monkeypatch.setattr("roboactions.remote_env.websocket.create_connection", fake_create_connection)

    env = RemoteEnv("CartPole-v1", api_key="rk_test")
    obs, info = env.reset()  # should return cached initial obs without sending 'reset'
    assert obs == [0.0]
    assert info == {}
    assert [m["op"] for m in stub.sent] == ["make"]

def test_reset_with_options_sends_wire(monkeypatch):
    frames = [make_ok_frame_discrete2(), reset_ok_frame([1.23])]
    stub = StubWS(frames=frames)
    monkeypatch.setattr(
        "roboactions.remote_env.websocket.create_connection", lambda url, header=None, timeout=None: stub
    )
    env = RemoteEnv("CartPole-v1", api_key="rk_test")
    # options prevents short-circuit; forces wire reset
    obs, info = env.reset(options={"foo": 1})
    assert obs == [1.23]
    assert info == {}
    assert [m["op"] for m in stub.sent] == ["make", "reset"]
    # Ensure options were forwarded
    assert stub.sent[1].get("options") == {"foo": 1}


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


def test_connect_retries_success_on_second_attempt(monkeypatch):
    calls = {"n": 0}
    frames = [make_ok_frame_discrete2()]
    stub = StubWS(frames=frames)

    def flaky_create_connection(url, header=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("transient network error")
        return stub

    monkeypatch.setattr("roboactions.remote_env.websocket.create_connection", flaky_create_connection)

    from roboactions.config import RetryConfig
    RemoteEnv("CartPole-v1", api_key="rk_test", retries=RetryConfig(max_attempts=3, backoff_factor=0.0))
    # ensure one make sent
    assert [m["op"] for m in stub.sent] == ["make"]
    # ensure it actually retried exactly once
    assert calls["n"] == 2


def test_connect_does_not_retry_on_401(monkeypatch):
    class HandshakeError(Exception):
        def __init__(self, status_code):
            super().__init__(f"status {status_code}")
            self.status_code = status_code

    calls = {"n": 0}

    def create_connection_401(url, header=None, timeout=None):
        calls["n"] += 1
        raise HandshakeError(401)

    monkeypatch.setattr("roboactions.remote_env.websocket.create_connection", create_connection_401)

    from roboactions.config import RetryConfig
    import pytest

    with pytest.raises(Exception):
        RemoteEnv("CartPole-v1", api_key="rk_test", retries=RetryConfig(max_attempts=3, backoff_factor=0.0))
    # should not retry past the first attempt
    assert calls["n"] == 1


def test_connect_retries_on_503_then_succeeds(monkeypatch):
    class HandshakeError(Exception):
        def __init__(self, status_code):
            super().__init__(f"status {status_code}")
            self.status_code = status_code

    calls = {"n": 0}
    frames = [make_ok_frame_discrete2()]
    stub = StubWS(frames=frames)

    def create_connection_503_then_ok(url, header=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise HandshakeError(503)
        return stub

    monkeypatch.setattr("roboactions.remote_env.websocket.create_connection", create_connection_503_then_ok)

    from roboactions.config import RetryConfig

    RemoteEnv("CartPole-v1", api_key="rk_test", retries=RetryConfig(max_attempts=3, backoff_factor=0.0))
    assert [m["op"] for m in stub.sent] == ["make"]
    assert calls["n"] == 3

