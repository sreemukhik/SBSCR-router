"""
SBSCR: Signature-Based Structural Complexity Routing
A lightweight LLM routing framework for sub-millisecond latency.
"""

__version__ = "0.1.0"

from sbscr.routers.sbscr import SBSCRRouter
from sbscr.core.models import ModelPool

__all__ = ["SBSCRRouter", "ModelPool"]
