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