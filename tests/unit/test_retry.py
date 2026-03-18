def test_retry():
    from src.core.retry import RetryManager

    r = RetryManager()
    assert r.execute_with_retry(lambda: 1) == 1

    import pytest

    def fail():
        e = ConnectionError("test")
        raise e

    with pytest.raises(Exception, match=".*"):
        r.execute_with_retry(fail)

    def fail_perm():
        e = ValueError("test")
        raise e

    with pytest.raises(ValueError, match=".*"):
        r.execute_with_retry(fail_perm)


def test_retry_2():
    import time

    import pytest

    from src.core.retry import RetryManager

    r = RetryManager()

    def slow():
        time.sleep(2)
        return 1

    with pytest.raises(Exception, match=".*"):
        r.execute_with_retry(slow, timeout=0.1)


def test_retry_misc():
    from src.core.retry import RetryManager

    r = RetryManager()

    import pytest

    with pytest.raises(Exception, match=".*"):
        r.execute_with_retry(lambda: 1 / 0, timeout=0.1, retries=1)


def test_retry_misc_2():
    from src.core.retry import RetryManager

    r = RetryManager()

    import contextlib

    import pytest

    def err():
        raise ConnectionError()

    for _ in range(6):
        with contextlib.suppress(Exception):
            r.execute_with_retry(err, breaker_name="test_circuit", retries=0)

    with pytest.raises(RuntimeError):
        r.execute_with_retry(lambda: 1, breaker_name="test_circuit")


def test_retry_misc_3():
    from src.core.retry import RetryManager

    r = RetryManager()

    def err():
        raise ConnectionError()

    import pytest

    with pytest.raises(Exception, match=".*"):
        r.execute_with_retry(err, breaker_name="test", retries=0)


def test_retry_misc_4():
    from src.core.retry import RetryManager

    r = RetryManager()

    def err():
        raise ConnectionError()

    import contextlib

    import pytest

    with pytest.raises(Exception, match=".*"):
        r.execute_with_retry(err, breaker_name="test2", retries=0)
    with contextlib.suppress(Exception):
        r.execute_with_retry(lambda: 1, breaker_name="test2")
