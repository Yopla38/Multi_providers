"""
AI Services Library
~~~~~~~~~~~~~~~~~~~

Unified interface for AI services (ComfyUI, Replicate, Local LLMs).

Basic usage:
    >>> from ai_services import media, llm
    >>> media.generate_image("A cat", "output.png")
    >>> llm.generate_text("Hello world")

:copyright: (c) 2024 by Jules.
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