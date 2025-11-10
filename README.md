# RoboActions Python SDK

Official Python SDK for the [RoboActions](https://www.roboactions.com) platform. It
provides a typed, ergonomic Python interface for interacting with deployed VLA policies, enabling
you to retrieve policy metadata, check health status, and run inference directly from Python
applications, scripts, or notebooks.

> **Status**: Pre-release (`0.1.0`). The surface area will grow as additional endpoints become
> publicly available. Follow the release notes before upgrading minor versions.

## Installation

```bash
pip install roboactions
```

## Quick Start

```python
from roboactions import RemotePolicy

# Create a policy client (automatically reads ROBOACTIONS_API_KEY from environment)
policy = RemotePolicy(policy_id="my-policy-id")

# Check policy health
status = policy.status()
print(f"Policy status: {status.value}")

# Get input/output feature schemas
input_features = policy.input_features()
output_features = policy.output_features()

# Discover existing policies
policies = RemotePolicy.list()
for summary in policies:
    print(f"{summary.policy_id}: {summary.display_name}")

# Run inference
observation = {
    "observation_image": image_array,
    "observation_state": state_array,
    "task": "pick up the cup"
}
action = policy.select_action(observation)
print(f"Predicted action: {action}")

# Predict multiple timesteps at once
action_chunk = policy.predict_action_chunk(observation)
print(f"Chunked action output: {action_chunk}")
```

## Authentication

All requests require an API key with access to the RoboActions workspace. Create and manage keys in
the RoboActions dashboard.

The SDK automatically reads the `ROBOACTIONS_API_KEY` environment variable:

```python
from roboactions import RemotePolicy

# Automatically uses ROBOACTIONS_API_KEY from environment
policy = RemotePolicy(policy_id="my-policy-id")
```

You can also provide the API key explicitly:

```python
from roboactions import RemotePolicy

policy = RemotePolicy(
    policy_id="my-policy-id",
    api_key="rk_live_abc123"  # Explicit API key
)
```

For security, it's recommended to use environment variables rather than hardcoding keys in your source code.

## Creating Policies

Use `RemotePolicy.create` to provision a policy directly from the SDK. The method handles authentication, and returns a ready-to-use client. Persist the returned `policy_id` if you plan to reconnect later. Set `compute_type` to `"CPU"` or `"GPU"` (default).

```python
from roboactions import RemotePolicy

policy = RemotePolicy.create(
    huggingface_model_id="my-org/my-robot-policy",
    compute_type="CPU",  # optional; defaults to "GPU"
    # api_key="rk_live_abc123",  # optional overrides
)

print(f"New policy: {policy.policy_id}")

# Reconnect elsewhere in your application with the policy ID
reconnected = RemotePolicy(policy_id=policy.policy_id)
```

## Core Features

### Policy Management

```python
# List all policies the current API key can access
from roboactions import RemotePolicy

policies = RemotePolicy.list()
for summary in policies:
    print(f"{summary.policy_id}: {summary.display_name}")

# Delete a policy you no longer need
policy = RemotePolicy(policy_id="my-policy-id")
delete_result = policy.delete()
print(f"Delete response: {delete_result}")
```

### Health Checks

```python
# Simple boolean check
if policy.ping():
    print("Policy is reachable")

# Detailed status with enum
from roboactions import PolicyStatus

status = policy.status()
if status == PolicyStatus.HEALTHY:
    print("Policy is ready for inference")
elif status == PolicyStatus.DEPLOYING:
    print("Policy is still deploying...")
```

### Wait for Deployment

```python
# Block until policy is deployed
final_status = policy.wait_until_deployed(interval=5.0)
if final_status == PolicyStatus.HEALTHY:
    print("Policy is now ready!")
```

### Reset Policy State

```python
# Reset the policy state (useful for stateful policies between episodes)
result = policy.reset()
if result.get("reset"):
    print("Policy reset successfully")
else:
    print(f"Reset failed: {result.get('error')}")
```

### Inference

```python
# Single action prediction
action = policy.select_action(observation)

# Action chunk prediction (for temporal policies)
action_chunk = policy.predict_action_chunk(observation)
```

### Sample Input Generation

```python
# Generate random sample inputs (useful for testing)
sample_inputs = policy.sample_observation()
action = policy.select_action(sample_inputs)
```

### Observation Shapes Cache

```python
# Retrieve observation tensor dimensions discovered via input_features
policy.input_features()
for key, shape in policy.observation_shapes:
    print(f"{key}: {shape}")
```

## Remote Gymnasium Environments

Interact with Gymnasium-compatible environments hosted via RoboActions over a secure WebSocket.

```python
from roboactions import remotegym

# Reads ROBOACTIONS_API_KEY from environment if not provided explicitly
env = remotegym.make("CartPole-v1", render_mode="rgb_array")

# Standard Gymnasium flow
obs, info = env.reset(seed=123)
for _ in range(100):
    action = env.action_space.sample()  # gymnasium.spaces.Space
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()  # numpy.ndarray (H, W, 3) when render_mode == "rgb_array"
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

Notes:
- `render()` returns a NumPy RGB array when `render_mode="rgb_array"`; otherwise returns `None`.
- The client pre-validates actions using the real Gymnasium `action_space`.
- If a step ends an episode, call `reset()` before the next `step()`.
- On connect, the server performs `reset(seed=seed)` and returns the initial `observation` and `info`.
- Additional properties exposed:
  - `observation_space`: Gymnasium space describing observations
  - `metadata`: Full `env.metadata` dict (e.g., `render_modes`, `render_fps`)
  - `spec`: `None` (standard Gymnasium attribute); raw mapping available via `server_spec`
  - `initial_observation`, `initial_info`: Values returned alongside `make_ok`
  - `render_mode`: The render mode passed at construction
  - `reward_range`: Default `(-inf, inf)` per Gymnasium

## Configuration

- **Retries (HTTP):** Configure automatic retries with `RemotePolicy(..., retries=RetryConfig())`
- **Timeouts (HTTP):** Set per-request timeouts via `policy.status(timeout=5.0)`
- **Retries (RemoteEnv WebSocket):** Enable connection retry/backoff with `RemoteEnv(..., retries=RetryConfig())` or via the helper `remotegym.make(..., retries=RetryConfig())`. Transient handshake failures (e.g., 408/429/5xx) are retried up to `max_attempts` with exponential backoff using `backoff_factor`. Authentication and other non-transient errors are not retried.
- **Timeouts (RemoteEnv WebSocket):** Control the WebSocket connect timeout with `RemoteEnv(..., timeout=...)` or `remotegym.make(..., timeout=...)`. Defaults: `60s` when constructing `RemoteEnv` directly; `10s` when using `remotegym.make` unless overridden.
- **Custom base URL:** Override `base_url` to point at staging or self-hosted deployments
- **Context Manager:** Use `with RemotePolicy(...) as policy:` for automatic session cleanup

## Exception Handling

```python
from roboactions import (
    RemotePolicy,
    RoboActionsError,
    AuthenticationError,
    RateLimitError
)

try:
    policy = RemotePolicy(policy_id="my-policy", api_key="invalid-key")
    action = policy.select_action(observation)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except RoboActionsError as e:
    print(f"API error: {e}")
```

## Development

1. Create a virtual environment and install dependencies with `pip install -e .[dev]`.
2. Run formatting and lint checks using `ruff check .`.
3. Execute the test suite via `pytest`.

## Releasing

1. Update `src/roboactions/_version.py`.
2. Build artifacts: `python -m build`.
3. Upload to PyPI (test or prod) using `twine upload dist/*`.

## License

Apache License 2.0 © RoboActions
