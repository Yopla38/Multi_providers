# 📦 Checklist pour Transformer en Bibliothèque Python

Voici **tout ce qu'il reste à faire** pour en faire une vraie bibliothèque professionnelle :

---

## 1. 🏗️ Structure du Package

```
ai-services-lib/
├── pyproject.toml              # ⭐ À CRÉER - Configuration moderne
├── setup.py                    # ⭐ À CRÉER - Installation
├── setup.cfg                   # ⭐ À CRÉER - Métadonnées
├── MANIFEST.in                 # ⭐ À CRÉER - Fichiers à inclure
├── README.md                   # ⭐ À CRÉER - Documentation principale
├── LICENSE                     # ⭐ À CRÉER - Licence (MIT recommandé)
├── CHANGELOG.md                # ⭐ À CRÉER - Historique des versions
├── requirements.txt            # ⭐ À CRÉER - Dépendances
├── requirements-dev.txt        # ⭐ À CRÉER - Dépendances dev
├── .gitignore                  # ⭐ À CRÉER
├── .github/                    # ⭐ À CRÉER - CI/CD
│   └── workflows/
│       └── tests.yml
│
├── ai_services/               # ⭐ RENOMMER services/ en ai_services/
│   ├── __init__.py           # ⭐ À AMÉLIORER - Version, exports
│   ├── __version__.py        # ⭐ À CRÉER
│   ├── config.yaml           # ✅ Existe
│   ├── exceptions.py         # ⭐ À CRÉER - Exceptions custom
│   ├── logging_config.py     # ⭐ À CRÉER - Config logging
│   ├── validators.py         # ⭐ À CRÉER - Validation config
│   ├── media_manager.py      # ✅ Existe
│   ├── llm_manager.py        # ✅ Existe
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py           # ⭐ À CRÉER - Classes abstraites
│   │   ├── utils.py          # ✅ Existe
│   │   ├── comfyui_providers.py    # ✅ Existe
│   │   ├── replicate_providers.py  # ✅ Existe
│   │   └── local_llm_providers.py  # ✅ Existe
│   │
│   └── cli/                  # ⭐ À CRÉER - Interface CLI
│       ├── __init__.py
│       └── main.py
│
├── tests/                     # ⭐ À CRÉER - Tests unitaires
│   ├── __init__.py
│   ├── conftest.py           # Fixtures pytest
│   ├── test_media_manager.py
│   ├── test_llm_manager.py
│   ├── test_providers/
│   │   ├── test_comfyui.py
│   │   ├── test_replicate.py
│   │   └── test_local_llm.py
│   └── fixtures/
│       └── test_images/
│
├── docs/                      # ⭐ À CRÉER - Documentation
│   ├── conf.py               # Sphinx config
│   ├── index.rst
│   ├── quickstart.rst
│   ├── api_reference.rst
│   ├── providers.rst
│   └── examples/
│
└── examples/                  # ⭐ À CRÉER - Exemples
    ├── basic_usage.py
    ├── advanced_workflow.py
    ├── custom_provider.py
    └── notebooks/
        └── tutorial.ipynb
```

---

## 2. 📝 Fichiers Essentiels à Créer

### **pyproject.toml** (Standard moderne)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-services-lib"
version = "0.1.0"
description = "Unified interface for AI services (ComfyUI, Replicate, Local LLMs)"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Votre Nom", email = "votre@email.com"}
]
keywords = ["ai", "llm", "comfyui", "stable-diffusion", "replicate"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "tenacity>=8.0",
    "requests>=2.31",
    "torch>=2.0",
    "pillow>=10.0",
    "instructor>=0.5.0",
    "llama-cpp-python>=0.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "black>=23.0",
    "flake8>=6.0",
    "mypy>=1.0",
    "sphinx>=7.0",
]

replicate = [
    "replicate>=0.20.0",
]

all = [
    "ai-services-lib[replicate,dev]",
]

[project.urls]
Homepage = "https://github.com/votre-nom/ai-services-lib"
Documentation = "https://ai-services-lib.readthedocs.io"
Repository = "https://github.com/votre-nom/ai-services-lib"
Changelog = "https://github.com/votre-nom/ai-services-lib/blob/main/CHANGELOG.md"

[project.scripts]
ai-services = "ai_services.cli.main:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["ai_services*"]

[tool.setuptools.package-data]
ai_services = ["config.yaml", "py.typed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=ai_services --cov-report=html --cov-report=term"

[tool.black]
line-length = 100
target-version = ['py39']

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

### **setup.py** (Pour compatibilité)

```python
"""
Setup script for ai-services-lib.
Uses pyproject.toml for configuration.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()
```

---

### **ai_services/__version__.py**

```python
"""Version information."""

__version__ = "0.1.0"
__author__ = "Votre Nom"
__email__ = "votre@email.com"
__license__ = "MIT"
__description__ = "Unified interface for AI services"
```

---

### **ai_services/__init__.py** (Amélioré)

```python
"""
AI Services Library
~~~~~~~~~~~~~~~~~~~

Unified interface for AI services (ComfyUI, Replicate, Local LLMs).

Basic usage:
    >>> from ai_services import media, llm
    >>> media.generate_image("A cat", "output.png")
    >>> llm.generate_text("Hello world")

:copyright: (c) 2024 by Votre Nom.
:license: MIT, see LICENSE for more details.
"""

from .__version__ import __version__, __author__, __email__
from .media_manager import MediaManager, media
from .llm_manager import LLMManager, llm
from .exceptions import (
    AIServicesError,
    ProviderError,
    ConfigurationError,
    GenerationError,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Main interfaces
    "media",
    "llm",
    "MediaManager",
    "LLMManager",
    
    # Exceptions
    "AIServicesError",
    "ProviderError",
    "ConfigurationError",
    "GenerationError",
]

# Configure logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

---

### **ai_services/exceptions.py**

```python
"""
Custom exceptions for ai-services-lib.
"""


class AIServicesError(Exception):
    """Base exception for all ai-services errors."""
    pass


class ProviderError(AIServicesError):
    """Error related to a specific provider."""
    
    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


class ConfigurationError(AIServicesError):
    """Error in configuration (YAML or secrets)."""
    pass


class GenerationError(AIServicesError):
    """Error during generation (image, video, text)."""
    
    def __init__(self, service: str, message: str, original_error: Exception = None):
        self.service = service
        self.original_error = original_error
        super().__init__(f"Generation failed in {service}: {message}")


class ModelNotFoundError(AIServicesError):
    """Model file not found."""
    pass


class ValidationError(AIServicesError):
    """Configuration validation failed."""
    pass
```

---

### **ai_services/validators.py**

```python
"""
Configuration validators.
"""
from pathlib import Path
from typing import Dict, Any
from .exceptions import ValidationError, ConfigurationError


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure.
    
    Raises:
        ValidationError: If configuration is invalid
    """
    # Check required sections
    required_sections = ["media_services", "llm_services", "providers"]
    for section in required_sections:
        if section not in config:
            raise ValidationError(f"Missing required section: {section}")
    
    # Validate provider references
    providers = config["providers"].keys()
    
    for service_type, services in config["media_services"].items():
        provider = services.get("provider")
        if provider not in providers:
            raise ValidationError(
                f"Unknown provider '{provider}' in media_services.{service_type}"
            )
    
    for service_type, services in config["llm_services"].items():
        provider = services.get("provider")
        if provider not in providers:
            raise ValidationError(
                f"Unknown provider '{provider}' in llm_services.{service_type}"
            )


def validate_paths(config: Dict[str, Any]) -> None:
    """
    Validate that required paths exist.
    
    Raises:
        ConfigurationError: If required paths don't exist
    """
    # ComfyUI paths
    if "comfyui" in config.get("providers", {}):
        base_path = config["providers"]["comfyui"].get("base_path")
        if base_path and not Path(base_path).exists():
            raise ConfigurationError(
                f"ComfyUI base path does not exist: {base_path}"
            )
    
    # Local LLM model paths
    for service_name, service_config in config.get("llm_services", {}).items():
        if "model_path" in service_config:
            model_path = service_config["model_path"]
            if not Path(model_path).exists():
                raise ConfigurationError(
                    f"Model not found for {service_name}: {model_path}"
                )
```

---

### **ai_services/logging_config.py**

```python
"""
Logging configuration.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the library.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger("ai_services")
    logger.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"ai_services.{name}")
```

---

### **README.md** (Complet)

```markdown
# 🤖 AI Services Library

> Unified interface for AI services: ComfyUI, Replicate, Local LLMs (DeepSeek, LLaVA)

[![PyPI version](https://badge.fury.io/py/ai-services-lib.svg)](https://badge.fury.io/py/ai-services-lib)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/votre-nom/ai-services-lib/workflows/tests/badge.svg)](https://github.com/votre-nom/ai-services-lib/actions)

## ✨ Features

- 🎨 **Image Generation**: Flux, SDXL via ComfyUI or Replicate
- ✏️ **Image Editing**: Context-aware modifications
- 🎬 **Video Generation**: Wan Video 2.2
- 🤖 **Local LLMs**: DeepSeek R1 with structured outputs (Pydantic)
- 👁️ **Multimodal**: LLaVA for image analysis
- ⚡ **Unified Interface**: Switch providers with one line
- 🔄 **Production-Ready**: Automatic retries, logging, error handling

## 🚀 Quick Start

### Installation

```bash
pip install ai-services-lib

# With optional dependencies
pip install ai-services-lib[replicate]  # Replicate support
pip install ai-services-lib[all]        # Everything
```

### Basic Usage

```python
from ai_services import media, llm

# Generate an image
media.generate_image(
    prompt="A serene Japanese garden",
    output_path="garden.png"
)

# Generate text
response = llm.generate_text("Write a haiku about coding")
print(response)

# Analyze an image
analysis = llm.analyze_image(
    prompt="What's in this image?",
    image_path="garden.png"
)
```

### Configuration

Create `~/.ai_services/config.yaml`:

```yaml
media_services:
  image_generation:
    provider: comfyui  # or 'replicate'
    workflow: flux_cinemat.json

llm_services:
  text_generation:
    provider: local_deepseek
    model_path: /path/to/model.gguf

providers:
  comfyui:
    base_path: /path/to/ComfyUI/
    url: http://127.0.0.1:8188
```

Create `~/.ai_services/secrets.env`:

```bash
REPLICATE_API_TOKEN=r8_your_token
```

## 📚 Documentation

- [Full Documentation](https://ai-services-lib.readthedocs.io)
- [API Reference](https://ai-services-lib.readthedocs.io/api)
- [Examples](./examples/)

## 🔧 Development

```bash
# Clone the repo
git clone https://github.com/votre-nom/ai-services-lib.git
cd ai-services-lib

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black ai_services tests
flake8 ai_services tests
```

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Credits

Built on top of:
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Instructor](https://github.com/jxnl/instructor)
```

---

### **MANIFEST.in**

```
include README.md
include LICENSE
include CHANGELOG.md
include requirements.txt
recursive-include ai_services *.yaml *.yml
recursive-include ai_services py.typed
recursive-exclude tests *
recursive-exclude docs *
```

---

### **.gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
ai_services/secrets.env
test_output/
*.log
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# OS
.DS_Store
Thumbs.db
```

---

## 3. ✅ Tests Unitaires Professionnels

### **tests/conftest.py**

```python
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
```

### **tests/test_validators.py**

```python
"""
Tests for configuration validators.
"""
import pytest
from ai_services.validators import validate_config, validate_paths
from ai_services.exceptions import ValidationError


def test_validate_config_valid(mock_config):
    """Test validation passes for valid config."""
    validate_config(mock_config)  # Should not raise


def test_validate_config_missing_section():
    """Test validation fails for missing section."""
    with pytest.raises(ValidationError, match="Missing required section"):
        validate_config({"providers": {}})


def test_validate_config_unknown_provider(mock_config):
    """Test validation fails for unknown provider."""
    mock_config["media_services"]["image_generation"]["provider"] = "unknown"
    
    with pytest.raises(ValidationError, match="Unknown provider"):
        validate_config(mock_config)
```

---

## 4. 🔄 CI/CD Pipeline

### **.github/workflows/tests.yml**

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with flake8
      run: |
        flake8 ai_services tests --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Type check with mypy
      run: |
        mypy ai_services
    
    - name: Test with pytest
      run: |
        pytest --cov=ai_services --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## 5. 📋 Récapitulatif des Actions

### ✅ Immédiat (Essentiel)

1. **Créer pyproject.toml** - Configuration moderne
2. **Créer exceptions.py** - Gestion d'erreurs propre
3. **Améliorer __init__.py** - Exports propres
4. **Créer README.md** - Documentation de base
5. **Créer requirements.txt** - Dépendances claires
6. **Renommer `services/` → `ai_services/`** - Nom de package valide

### 🔧 Court terme (Important)

7. **Ajouter validators.py** - Validation robuste
8. **Créer tests unitaires** - Au moins 50% coverage
9. **Ajouter logging_config.py** - Logs professionnels
10. **Créer .gitignore** - Éviter les commits inutiles
11. **Ajouter type hints partout** - `mypy` compliant
12. **Créer CHANGELOG.md** - Suivi des versions

### 📚 Moyen terme (Polissage)

13. **Documentation Sphinx** - API Reference
14. **CLI interface** - `ai-services generate-image ...`
15. **GitHub Actions** - Tests auto
16. **Exemples avancés** - Notebooks Jupyter
17. **Publish sur PyPI** - `pip install ai-services-lib`

### 🚀 Long terme (Évolution)

18. **Plugin system** - Providers externes
19. **Cache intelligent** - Éviter régénérations
20. **Monitoring** - Prometheus metrics
21. **Web UI** - Interface Gradio/Streamlit
22. **Docker images** - Déploiement facile

---

## 6. 🎯 Commandes pour Commencer

```bash
# 1. Restructurer
mv services ai_services
touch ai_services/py.typed  # Pour mypy

# 2. Créer les fichiers essentiels
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-services-lib"
version = "0.1.0"
dependencies = [
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "tenacity>=8.0",
]
EOF

cat > setup.py << 'EOF'
from setuptools import setup
setup()
EOF

# 3. Créer requirements.txt
cat > requirements.txt << 'EOF'
pyyaml>=6.0
pydantic>=2.0
tenacity>=8.0
requests>=2.31
torch>=2.0
pillow>=10.0
instructor>=0.5.0
llama-cpp-python>=0.2.0
EOF

# 4. Test d'installation
pip install -e .

# 5. Test d'import
python -c "from ai_services import media, llm; print('✅ OK')"
```

---
