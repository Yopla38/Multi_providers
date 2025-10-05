"""
Interface simplifiée pour la bibliothèque de services AI.
"""
from .media_manager import media
from .llm_manager import llm

__all__ = ['media', 'llm']

__version__ = "1.0.0"