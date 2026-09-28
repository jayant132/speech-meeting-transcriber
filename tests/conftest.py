import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (real models)")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def sample_audio_path():
    return "samples/demo.mp3"