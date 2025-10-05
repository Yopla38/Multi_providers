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