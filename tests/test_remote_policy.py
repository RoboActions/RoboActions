import pytest

from roboactions.policy import PolicyStatus, RemotePolicy


class StubHttpClient:
    def __init__(self, config, session=None):
        self.config = config
        self._config = config
        self.session = session
        self.requests = []
        self.responses = []

    def enqueue(self, payload):
        self.responses.append(payload)

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if not self.responses:
            raise AssertionError("No stubbed response available")
        return self.responses.pop(0)

    def close(self):
        pass


class TimeStub:
    def __init__(self, start=0.0):
        self._now = start
        self.sleeps = []

    def monotonic(self):
        return self._now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self._now += seconds


def test_remote_policy_requires_identifiers():
    with pytest.raises(ValueError):
        RemotePolicy(policy_id="", api_key="rk_test")

    with pytest.raises(ValueError):
        RemotePolicy(policy_id="pol-123", api_key="")


def test_remote_policy_calls_expected_endpoints(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-123", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue({"status": "HEALTHY"})
    http.enqueue(
        {
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
                "action.mask": {"type": "ACTION", "shape": [1]},
            }
        }
    )
    http.enqueue({"features": ["risk_score"]})

    status = policy.status()
    inputs = policy.input_features()
    outputs = policy.output_features()

    assert status is PolicyStatus.HEALTHY
    assert inputs == {
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
            "action.mask": {"type": "ACTION", "shape": [1]},
        }
    }
    assert outputs == {"features": ["risk_score"]}
    assert policy.observation_shapes == [
        ("observation.state", (6,)),
        ("observation.images.camera1", (3, 256, 256)),
    ]

    assert http.requests[0] == ("GET", "/v1/healthz", {"timeout": None})
    assert http.requests[1] == ("GET", "/v1/policy/input_features", {"timeout": None})
    assert http.requests[2] == ("GET", "/v1/policy/output_features", {"timeout": None})

    headers = http.config.headers_with_auth()
    assert headers["x-policy-id"] == "pol-123"
    assert headers["Authorization"] == "Bearer rk_test"
    assert http.config.base_url == "https://api.roboactions.com"


def test_input_features_observation_shapes_cache(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-001", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    assert policy.observation_shapes == []

    http.enqueue(
        {
            "input_features": {
                "observation.state": {"shape": [4]},
                "observation.images.depth": {"shape": [1, 128, 128]},
            }
        }
    )

    payload = policy.input_features()
    assert payload["input_features"]["observation.state"]["shape"] == [4]

    cached = policy.observation_shapes
    cached.append(("observation.state", (1,)))  # mutate returned list copy

    assert cached != policy.observation_shapes
    assert policy.observation_shapes == [
        ("observation.state", (4,)),
        ("observation.images.depth", (1, 128, 128)),
    ]


def test_select_and_predict_action_use_msgpack(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-xyz", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    pack_calls = []
    unpack_calls = []

    def fake_packb(obj):
        pack_calls.append(obj)
        return f"packed-{len(pack_calls)}".encode()

    def fake_unpackb(data):
        unpack_calls.append(data)
        return {"decoded": data.decode()}

    monkeypatch.setattr("roboactions.policy.packb", fake_packb)
    monkeypatch.setattr("roboactions.policy.unpackb", fake_unpackb)

    http.enqueue(b"select-response")
    http.enqueue(b"predict-response")

    observation = {"observation.state": [1, 2, 3]}

    select = policy.select_action(observation)
    predict = policy.predict_action_chunk(observation)

    assert select == {"decoded": "select-response"}
    assert predict == {"decoded": "predict-response"}

    assert pack_calls == [observation, observation]
    assert unpack_calls == [b"select-response", b"predict-response"]

    assert len(http.requests) == 2
    method_one, path_one, kwargs_one = http.requests[0]
    method_two, path_two, kwargs_two = http.requests[1]

    assert (method_one, path_one) == ("POST", "/v1/policy/select_action")
    assert (method_two, path_two) == ("POST", "/v1/policy/predict_action_chunk")

    for kwargs in (kwargs_one, kwargs_two):
        headers = kwargs["headers"]
        assert headers["Content-Type"] == "application/msgpack"
        assert headers["Accept"] == "application/msgpack"
        assert headers["Authorization"] == "Bearer rk_test"
        assert kwargs["expect_json"] is False
        assert kwargs["data"].startswith(b"packed-")

def test_sample_observation_uses_cached_shapes(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-xyz", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue(
        {
            "input_features": {
                "observation.state": {"shape": [6]},
                "observation.images.camera1": {"shape": [3, 256, 256]},
                "action.mask": {"shape": [1]},
            }
        }
    )

    class TorchStub:
        float32 = "float32"

        def __init__(self):
            self.calls = []

        def rand(self, shape, dtype=None):
            self.calls.append((shape, dtype))
            return {"shape": shape, "dtype": dtype}

    torch_stub = TorchStub()
    monkeypatch.setattr("roboactions.policy.torch", torch_stub)

    samples = policy.sample_observation()
    assert http.requests == [("GET", "/v1/policy/input_features", {"timeout": None})]
    assert set(samples.keys()) == {
        "observation.state",
        "observation.images.camera1",
        "task",
    }
    assert samples["observation.state"] == {"shape": (6,), "dtype": "float32"}
    assert samples["observation.images.camera1"] == {"shape": (3, 256, 256), "dtype": "float32"}
    assert samples["task"] == "Random Task"
    assert torch_stub.calls == [
        ((6,), "float32"),
        ((3, 256, 256), "float32"),
    ]

    torch_stub.calls.clear()
    samples_again = policy.sample_observation()
    assert set(samples_again.keys()) == set(samples.keys())
    assert samples_again["task"] == "Random Task"
    assert http.requests == [("GET", "/v1/policy/input_features", {"timeout": None})]
    assert torch_stub.calls == [
        ((6,), "float32"),
        ((3, 256, 256), "float32"),
    ]

def test_remote_policy_status_unknown(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-123", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue({})
    http.enqueue({"status": "invalid"})

    assert policy.status() is PolicyStatus.UNKNOWN
    assert policy.status() is PolicyStatus.UNKNOWN


def test_wait_until_deployed_polls_until_success(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)
    time_stub = TimeStub()
    monkeypatch.setattr("roboactions.policy.time", time_stub)

    policy = RemotePolicy(policy_id="pol-123", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue({"status": "DEPLOYING"})
    http.enqueue({"status": "HEALTHY"})

    status = policy.wait_until_deployed(interval=1.0)

    assert status is PolicyStatus.HEALTHY
    assert len(http.requests) == 2
    assert http.requests[0] == ("GET", "/v1/healthz", {"timeout": None})
    assert http.requests[1] == ("GET", "/v1/healthz", {"timeout": None})
    assert time_stub.sleeps == [1.0]


def test_wait_until_deployed_returns_non_deploying_status(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)
    time_stub = TimeStub()
    monkeypatch.setattr("roboactions.policy.time", time_stub)

    policy = RemotePolicy(policy_id="pol-123", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue({"status": "DEPLOYING"})
    http.enqueue({"status": "STOPPED"})

    status = policy.wait_until_deployed(interval=0.5)

    assert status is PolicyStatus.STOPPED
    assert len(http.requests) == 2
    assert http.requests[0] == ("GET", "/v1/healthz", {"timeout": None})
    assert http.requests[1] == ("GET", "/v1/healthz", {"timeout": None})
    assert time_stub.sleeps == [0.5]


def test_wait_until_deployed_requires_positive_interval(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-123", api_key="rk_test")

    with pytest.raises(ValueError):
        policy.wait_until_deployed(interval=0.0)


def test_reset_returns_success(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-123", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue({"reset": True})
    result = policy.reset()

    assert result == {"reset": True}
    assert http.requests[0] == ("GET", "/v1/policy/reset", {"timeout": None})


def test_reset_returns_failure_with_error(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-123", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue({"reset": False, "error": "Policy is not in a resettable state"})
    result = policy.reset()

    assert result == {"reset": False, "error": "Policy is not in a resettable state"}
    assert http.requests[0] == ("GET", "/v1/policy/reset", {"timeout": None})
