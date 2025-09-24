"""
API package initialization for the Multimodal Processing FastAPI backend.
Exposes the FastAPI application instance and the app factory for external use (e.g., ASGI servers, tests).
"""

# PUBLIC_INTERFACE
def __all__():
    """
    Public exports from the api package.
    Returns:
        list[str]: The names that should be imported when using `from api import *`.
    """
    return ["app", "get_app"]

# Re-export app and factory for consumers
from .main import app, get_app  # noqa: E402,F401
