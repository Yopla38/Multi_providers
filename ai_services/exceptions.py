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