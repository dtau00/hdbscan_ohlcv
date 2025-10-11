"""HDBSCAN OHLCV Pattern Discovery

GPU-accelerated HDBSCAN clustering for OHLCV trading data with CPU fallback.
"""

from .gpu_utils import detect_compute_backend, get_backend_info, BackendInfo
from .data_loader import OHLCVDataLoader, generate_sample_ohlcv
from .config import Config

__version__ = "0.1.0"
__author__ = "HDBSCAN OHLCV Team"

__all__ = [
    # GPU/Backend utilities
    "detect_compute_backend",
    "get_backend_info",
    "BackendInfo",
    # Data loading
    "OHLCVDataLoader",
    "generate_sample_ohlcv",
    # Configuration
    "Config",
]
