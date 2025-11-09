import pytest

from roboactions.policy import PolicyStatus, PolicySummary, RemotePolicy


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


def test_create_policy_returns_remote_policy(monkeypatch):
    clients = []

    def http_factory(config, session=None):
        client = StubHttpClient(config, session=session)
        if not clients:
            client.enqueue({"policy_id": "pol-new"})
        clients.append(client)
        return client

    monkeypatch.setattr("roboactions.policy.HttpClient", http_factory)

    policy = RemotePolicy.create(
        huggingface_model_id="org/model",
        api_key="rk_test",
        default_headers={"X-Custom": "42"},
    )

    assert isinstance(policy, RemotePolicy)
    assert policy.policy_id == "pol-new"
    assert len(clients) == 2

    create_client = clients[0]
    policy_client = clients[1]

    method, path, kwargs = create_client.requests[0]
    assert (method, path) == ("POST", "/v1/policy/create")
    assert kwargs["json_body"] == {"huggingface_model_id": "org/model", "compute_type": "GPU"}
    assert kwargs["timeout"] == create_client.config.timeout

    create_headers = create_client.config.headers_with_auth()
    assert create_headers["Authorization"] == "Bearer rk_test"
    assert "x-policy-id" not in create_headers
    assert create_client.config.default_headers["Accept"] == "application/json"
    assert create_client.config.default_headers["X-Custom"] == "42"

    policy_headers = policy_client.config.headers_with_auth()
    assert policy_headers["Authorization"] == "Bearer rk_test"
    assert policy_headers["x-policy-id"] == "pol-new"
    assert policy_headers["X-Custom"] == "42"
    assert policy_client.config.base_url == "https://api.roboactions.com"


def test_policy_summary_from_payload_requires_policy_id():
    with pytest.raises(ValueError, match="policy_id"):
        PolicySummary.from_payload({"display_name": "Unnamed"})


def test_remote_policy_list_returns_policy_summaries(monkeypatch):
    clients = []

    def http_factory(config, session=None):
        client = StubHttpClient(config, session=session)
        client.enqueue(
            [
                {"policy_id": "pol-1", "display_name": "Policy One"},
                {"policy_id": "pol-2", "display_name": "Policy Two"},
            ]
        )
        clients.append(client)
        return client

    monkeypatch.setattr("roboactions.policy.HttpClient", http_factory)

    summaries = RemotePolicy.list(api_key="rk_test")

    assert summaries == [
        PolicySummary(policy_id="pol-1", display_name="Policy One"),
        PolicySummary(policy_id="pol-2", display_name="Policy Two"),
    ]

    assert len(clients) == 1
    method, path, kwargs = clients[0].requests[0]
    assert (method, path) == ("GET", "/v1/policy/list")
    assert kwargs["timeout"] == clients[0].config.timeout
    headers = clients[0].config.default_headers
    assert headers["Accept"] == "application/json"
    assert headers["Content-Type"] == "application/json"


def test_remote_policy_list_rejects_non_list_response(monkeypatch):
    def http_factory(config, session=None):
        client = StubHttpClient(config, session=session)
        client.enqueue({"policy_id": "not-a-list"})
        return client

    monkeypatch.setattr("roboactions.policy.HttpClient", http_factory)

    with pytest.raises(TypeError, match="Expected list response"):
        RemotePolicy.list(api_key="rk_test")


def test_remote_policy_list_requires_mapping_items(monkeypatch):
    def http_factory(config, session=None):
        client = StubHttpClient(config, session=session)
        client.enqueue(
            [
                {"policy_id": "pol-1", "display_name": "Policy One"},
                "invalid-item",
            ]
        )
        return client

    monkeypatch.setattr("roboactions.policy.HttpClient", http_factory)

    with pytest.raises(TypeError, match="Expected mapping items"):
        RemotePolicy.list(api_key="rk_test")


def test_create_policy_requires_policy_id(monkeypatch):
    def http_factory(config, session=None):
        client = StubHttpClient(config, session=session)
        client.enqueue({})
        return client

    monkeypatch.setattr("roboactions.policy.HttpClient", http_factory)

    with pytest.raises(ValueError, match="policy_id"):
        RemotePolicy.create(api_key="rk_test", huggingface_model_id="org/model")


def test_create_policy_requires_model_id(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    with pytest.raises(ValueError, match="huggingface_model_id"):
        RemotePolicy.create(huggingface_model_id="", api_key="rk_test")


def test_create_policy_allows_compute_type_override(monkeypatch):
    clients = []

    def http_factory(config, session=None):
        client = StubHttpClient(config, session=session)
        if not clients:
            client.enqueue({"policy_id": "pol-cpu"})
        clients.append(client)
        return client

    monkeypatch.setattr("roboactions.policy.HttpClient", http_factory)

    policy = RemotePolicy.create(
        huggingface_model_id="org/model",
        compute_type="cpu",
        api_key="rk_test",
    )

    assert policy.policy_id == "pol-cpu"

    method, path, kwargs = clients[0].requests[0]
    assert (method, path) == ("POST", "/v1/policy/create")
    assert kwargs["json_body"] == {"huggingface_model_id": "org/model", "compute_type": "CPU"}


def test_create_policy_rejects_invalid_compute_type(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    with pytest.raises(ValueError, match="either 'CPU' or 'GPU'"):
        RemotePolicy.create(huggingface_model_id="org/model", compute_type="TPU", api_key="rk_test")


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


def test_delete_policy_posts_to_delete_endpoint(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-del", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue({"deleted": True})
    result = policy.delete()

    assert result == {"deleted": True}
    assert http.requests == [
        (
            "POST",
            "/v1/policy/delete",
            {
                "json_body": {"policy_id": "pol-del"},
                "timeout": None,
            },
        )
    ]


def test_delete_policy_requires_mapping_response(monkeypatch):
    monkeypatch.setattr("roboactions.policy.HttpClient", StubHttpClient)

    policy = RemotePolicy(policy_id="pol-del", api_key="rk_test")
    http = policy._http  # type: ignore[attr-defined]

    http.enqueue(["not-a-mapping"])

    with pytest.raises(TypeError, match="mapping response"):
        policy.delete()
