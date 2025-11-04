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

# Run inference
observation = {
    "observation_image": image_array,
    "observation_state": state_array,
    "task": "pick up the cup"
}
action = policy.select_action(observation)
print(f"Predicted action: {action}")
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

## Core Features

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

## Configuration

- **Retries:** Configure automatic retries with `RemotePolicy(..., retries=RetryConfig())`
- **Timeouts:** Set per-request timeouts via `policy.status(timeout=5.0)`
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
