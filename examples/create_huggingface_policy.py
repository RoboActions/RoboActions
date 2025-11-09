"""Example showing how to create a RoboActions policy from a Hugging Face model."""

import os

from roboactions import PolicyStatus, RemotePolicy


def main() -> None:
    api_key = os.environ.get("ROBOACTIONS_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOACTIONS_API_KEY before creating a policy")

    huggingface_model_id = os.environ.get("ROBOACTIONS_HF_MODEL_ID")
    if not huggingface_model_id:
        raise SystemExit("Set ROBOACTIONS_HF_MODEL_ID to the Hugging Face model you want to deploy")

    compute_type = os.environ.get("ROBOACTIONS_COMPUTE_TYPE", "GPU")

    print("Creating policy from Hugging Face model...")

    with RemotePolicy.create(
        huggingface_model_id=huggingface_model_id,
        compute_type=compute_type,
    ) as policy:
        print(f"Created policy: {policy.policy_id}")
        print(f"Compute type: {compute_type.upper()}")

        all_policies = RemotePolicy.list(api_key=api_key)
        print("Policies accessible to this API key:")
        for summary in all_policies:
            print(f"- {summary.policy_id}: {summary.display_name}")

        status = policy.status()
        print(f"Initial status: {status.value}")

        if status == PolicyStatus.DEPLOYING:
            print("Waiting for deployment...")
            status = policy.wait_until_deployed(interval=5.0)
            print(f"Deployment complete with status: {status.value}")

        input_features = policy.input_features()
        print(f"Input feature keys: {list(input_features['input_features'].keys())}")

    print("Reconnecting to the new policy instance...")
    with RemotePolicy(policy_id=policy.policy_id) as reconnected:
        print(f"Reconnected policy status: {reconnected.status().value}")

        delete_after_demo = os.environ.get("ROBOACTIONS_DELETE_AFTER_CREATE", "").lower()
        if delete_after_demo in {"1", "true", "yes"}:
            print("Deleting policy after demo...")
            delete_result = reconnected.delete()
            print(f"Delete response: {delete_result}")


if __name__ == "__main__":
    main()

