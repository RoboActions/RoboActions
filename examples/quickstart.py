"""Minimal sample showing how to use the RoboActions RemotePolicy."""

import os

from roboactions import PolicyStatus, RemotePolicy


def main() -> None:
    # RemotePolicy automatically reads ROBOACTIONS_API_KEY from environment
    # You can also pass api_key explicitly: RemotePolicy(policy_id="...", api_key="...")
    policy_id = os.environ.get("ROBOACTIONS_POLICY_ID")
    if not policy_id:
        raise SystemExit("Set ROBOACTIONS_POLICY_ID before running the example")

    # Create a remote policy client with context manager for automatic cleanup
    with RemotePolicy(policy_id=policy_id) as policy:
        # Check connectivity
        if not policy.ping():
            raise SystemExit("Unable to reach RoboActions API or policy not found")

        # Check policy status
        status = policy.status()
        print(f"Policy status: {status.value}")

        # Wait for deployment if necessary
        if status == PolicyStatus.DEPLOYING:
            print("Policy is deploying, waiting...")
            status = policy.wait_until_deployed(interval=5.0)
            print(f"Policy ready with status: {status.value}")

        if status != PolicyStatus.HEALTHY:
            raise SystemExit(f"Policy is not healthy: {status.value}")

        # Retrieve feature schemas
        input_features = policy.input_features()
        output_features = policy.output_features()
        print(f"Input features: {list(input_features["input_features"].keys())}")
        print(f"Output features: {list(output_features["output_features"].keys())}")

        # Reset policy state (optional - useful for stateful policies)
        reset_result = policy.reset()
        if reset_result.get("reset"):
            print("\nPolicy state reset successfully")
        else:
            print(f"\nPolicy reset failed: {reset_result.get('error', 'Unknown error')}")

        # Generate sample inputs and run inference
        print("\nRunning sample inference...")
        sample_obs = policy.sample_observation()
        action = policy.select_action(sample_obs)
        print(f"Predicted action shape: {action.shape if hasattr(action, 'shape') else type(action)}")


if __name__ == "__main__":
    main()
