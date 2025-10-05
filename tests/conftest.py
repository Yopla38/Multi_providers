"""
Pytest configuration and fixtures.
"""
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "media_services": {
            "image_generation": {
                "provider": "comfyui",
                "workflow": "test.json"
            }
        },
        "llm_services": {
            "text_generation": {
                "provider": "local_deepseek",
                "model_path": "/fake/path"
            }
        },
        "providers": {
            "comfyui": {
                "base_path": "/fake/comfyui",
                "url": "http://localhost:8188"
            }
        }
    }


@pytest.fixture
def mock_secrets():
    """Mock secrets for testing."""
    return {
        "REPLICATE_API_TOKEN": "r8_test_token"
    }