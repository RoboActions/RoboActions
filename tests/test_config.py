from roboactions.config import ClientConfig, RetryConfig


def test_client_config_sets_authorization_header():
    config = ClientConfig(api_key="rk_test_123")
    headers = config.headers_with_auth()
    assert headers["Authorization"] == "Bearer rk_test_123"


def test_retry_config_should_retry_only_selected_status_codes():
    retries = RetryConfig(enabled=True, max_attempts=2)
    assert retries.should_retry(500, attempt=0) is True
    assert retries.should_retry(404, attempt=0) is False
    assert retries.should_retry(500, attempt=2) is False
