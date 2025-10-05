# 🤖 AI Services Library

> A unified interface for seamless interaction with a variety of AI services, including ComfyUI for media generation, Replicate for cloud-based models, and local LLMs like DeepSeek and LLaVA.

[![PyPI version](https://badge.fury.io/py/ai-services-lib.svg)](https://badge.fury.io/py/ai-services-lib)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎨 **Advanced Image Generation**: Create and edit images using local ComfyUI workflows or cloud-based Replicate models.
- 🎬 **Automated Video Production**: Generate video sequences with the powerful Wan Video 2.2 workflow in ComfyUI.
- 🤖 **Local Language Models**: Run powerful LLMs like DeepSeek R1 for text generation with structured Pydantic outputs and LLaVA for multimodal analysis—all on your own hardware.
- ☁️ **Cloud-Based Flexibility**: Easily switch to Replicate for image and video generation without changing your code.
- 🔧 **Centralized Configuration**: Manage all service providers, model paths, and parameters from a single `config.yaml` file.
- 🔄 **Resilient & Production-Ready**: Features automatic retries, clear logging, and robust error handling.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

1.  **Main Configuration**: Modify `ai_services/config.yaml` to match your local setup. This file is the central hub for defining service providers, model paths, and other parameters.

2.  **Secrets**: Create a `secrets.env` file in the `ai_services/` directory for your API keys. **This file is git-ignored by default.**

    ```bash
    # ai_services/secrets.env
    REPLICATE_API_TOKEN=r8_your_replicate_token
    ```

## 🛠️ Functionalities

This library provides a unified interface for both local and cloud-based AI services.

### Local Services

Harness the power of your own hardware for complete control and privacy.

-   **ComfyUI**:
    -   **Image Generation & Editing**: Execute any ComfyUI workflow for generating or modifying images. The system is designed to be flexible, allowing you to map nodes for dynamic parameter changes.
    -   **Video Generation**: Automates the execution of complex video generation workflows like "Wan Video 2.2", managing frame-by-frame generation, coherence checks, and final video assembly.
-   **Local LLMs**:
    -   **DeepSeek R1**: A powerful text generation model for complex reasoning and instruction-following tasks. It supports structured outputs using Pydantic for predictable and reliable results.
    -   **LLaVA**: A multimodal model that can analyze images and answer questions about them, enabling sophisticated visual understanding capabilities.

### Replicate

Leverage a vast library of cloud-based AI models with minimal effort.

-   **Image Generation**: Access state-of-the-art image models like SDXL Lightning for near-instantaneous image creation.
-   **Video Generation**: Utilize models like Stable Video Diffusion to generate short video clips from text or image prompts.
-   **Easy Switching**: You can switch from a local ComfyUI workflow to a Replicate model by changing a single line in `config.yaml`, allowing for easy comparison and flexible deployment.

## ⚙️ Configuration Example (`config.yaml`)

The `config.yaml` file, located in the `ai_services/` directory, is the heart of the library's configuration.

```yaml
# ------------------------------------------------------------------------------
# 1. ASSIGNATION ET CONFIGURATION DES SERVICES
# ------------------------------------------------------------------------------
# Defines the provider, workflow, and node mapping for each service.
# Workflows are relative to the 'workflows' directory at the project root.
# ------------------------------------------------------------------------------
services:
  image_generation:
    provider: comfyui
    workflow: "flux_cinemat.json"
    node_mapping:
      prompt_node_id: 6
      save_node_id: 9
      # ... other nodes

  video_generation:
    provider: comfyui
    workflow: "WAN 2.2_loop_complete_api.json"
    node_mapping:
      input_image_node_id: 433
      prompt_node_id: 59
      # ... other nodes

  text_generation:
    provider: local_deepseek

  multimodal:
    provider: local_llava

# ------------------------------------------------------------------------------
# 2. CONFIGURATION DES FOURNISSEURS (PROVIDERS)
# ------------------------------------------------------------------------------
providers:
  comfyui:
    base_path: "/path/to/your/ComfyUI"
    url: "http://127.0.0.1:8188"

  replicate:
    default_models:
      image: "bytedance/sdxl-lightning-4step"
      video: "stability-ai/stable-video-diffusion"

  local_deepseek:
    model_path: "/path/to/your/deepseek.gguf"
    n_ctx: 8192
    temperature: 0.6
    max_tokens: 8000

  local_llava:
    model_path: "/path/to/your/llava.gguf"
    clip_path: "/path/to/your/llava_clip.gguf"
    n_ctx: 4096
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

This library is built on top of several incredible open-source projects:
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Instructor](https://github.com/jxnl/instructor)