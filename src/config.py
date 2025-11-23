"""Configuration module for reading environment variables and defaults."""

import os
from typing import Optional


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your environment or Apify actor settings."
        )
    return api_key


def get_default_model() -> str:
    """Get default OpenAI model."""
    return "gpt-4o-mini"

