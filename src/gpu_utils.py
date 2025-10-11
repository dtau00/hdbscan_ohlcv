"""GPU Detection and Compute Backend Configuration

This module detects available compute backend (GPU/CPU) and provides
appropriate HDBSCAN module selection with fallback support.
"""

import logging
import os
from dataclasses import dataclass
from typing import Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Backend detection cache
_BACKEND_CACHE: Optional[Tuple[str, Any]] = None


@dataclass
class BackendInfo:
    """Structured information about the compute backend."""

    backend_type: str  # 'gpu' or 'cpu'
    module_name: str
    cuda_version: Optional[int] = None
    device_name: Optional[str] = None
    device_memory_gb: Optional[float] = None
    free_memory_gb: Optional[float] = None

    def __str__(self) -> str:
        lines = [f"Backend: {self.backend_type.upper()}", f"Module: {self.module_name}"]
        if self.backend_type == 'gpu' and self.cuda_version:
            lines.append(f"CUDA Version: {self.cuda_version}")
            if self.device_name:
                lines.append(f"Device: {self.device_name}")
            if self.device_memory_gb and self.free_memory_gb:
                lines.append(f"Memory: {self.free_memory_gb:.2f}GB / {self.device_memory_gb:.2f}GB")
        return "\n".join(lines)


def detect_compute_backend(force_refresh: bool = False, min_gpu_memory_gb: Optional[float] = None) -> Tuple[str, Any]:
    """
    Detects available compute backend (GPU/CPU) and returns appropriate module.

    Attempts to import GPU-accelerated libraries (cupy, cuml) first.
    Falls back to CPU-based hdbscan if GPU libraries are unavailable.
    Results are cached to avoid repeated detection overhead.

    Args:
        force_refresh: If True, bypass cache and re-detect backend
        min_gpu_memory_gb: Minimum free GPU memory required (GB).
                          If None, reads from MIN_GPU_MEMORY_GB env var (default: 1.0)

    Returns:
        tuple: (backend_type: str, backend_module)
            - backend_type: 'gpu' or 'cpu'
            - backend_module: cuml.cluster or hdbscan module

    Examples:
        >>> backend_type, backend_module = detect_compute_backend()
        >>> if backend_type == 'gpu':
        ...     clusterer = backend_module.HDBSCAN(min_cluster_size=5)

        >>> # Set via environment variable
        >>> os.environ['MIN_GPU_MEMORY_GB'] = '2.0'
        >>> backend = detect_compute_backend(force_refresh=True)
    """
    global _BACKEND_CACHE

    # Get min_gpu_memory_gb from env var if not provided
    if min_gpu_memory_gb is None:
        min_gpu_memory_gb = float(os.getenv('MIN_GPU_MEMORY_GB', '1.0'))

    # Return cached result if available
    if _BACKEND_CACHE is not None and not force_refresh:
        logger.debug("Using cached backend configuration")
        return _BACKEND_CACHE

    # Try to import GPU libraries
    try:
        import cupy as cp

        # Check if CUDA is actually available
        if not cp.cuda.is_available():
            logger.info("CuPy installed but CUDA not available, falling back to CPU")
            raise ImportError("CUDA not available")

        # Check GPU memory availability
        try:
            device = cp.cuda.Device()
            free_mem, total_mem = device.mem_info
            free_gb = free_mem / 1e9

            if free_gb < min_gpu_memory_gb:
                logger.warning(
                    f"Low GPU memory ({free_gb:.2f}GB < {min_gpu_memory_gb}GB), "
                    f"falling back to CPU"
                )
                raise ImportError("Insufficient GPU memory")

            logger.info(f"GPU memory: {free_gb:.2f}GB free of {total_mem/1e9:.2f}GB total")
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}, proceeding anyway")

        # Try to import cuML HDBSCAN
        try:
            import cuml.cluster
            logger.info(f"GPU backend detected: CUDA {cp.cuda.runtime.runtimeGetVersion()}")
            logger.info(f"Using cuML HDBSCAN for GPU acceleration")
            _BACKEND_CACHE = ('gpu', cuml.cluster)
            return _BACKEND_CACHE

        except ImportError as e:
            logger.info(f"cuML not available ({e}), falling back to CPU")
            raise ImportError("cuML not available")

    except ImportError:
        # Fallback to CPU
        try:
            import hdbscan
            logger.info("Using CPU backend with standard hdbscan library")
            _BACKEND_CACHE = ('cpu', hdbscan)
            return _BACKEND_CACHE

        except ImportError as e:
            logger.error("Neither GPU (cuml) nor CPU (hdbscan) libraries available")
            raise ImportError(
                "No HDBSCAN implementation found. Please install either:\n"
                "  - GPU: pip install cuml-cu11 cupy-cuda11x\n"
                "  - CPU: pip install hdbscan"
            ) from e


def get_backend_info(backend_type: str, backend_module: Any) -> BackendInfo:
    """
    Get detailed information about the compute backend.

    Args:
        backend_type: Type of backend ('gpu' or 'cpu')
        backend_module: The HDBSCAN module being used

    Returns:
        BackendInfo object containing structured backend information
    """
    info = BackendInfo(
        backend_type=backend_type,
        module_name=backend_module.__name__
    )

    if backend_type == 'gpu':
        try:
            import cupy as cp
            device = cp.cuda.Device()
            free_mem, total_mem = device.mem_info

            info.cuda_version = cp.cuda.runtime.runtimeGetVersion()
            info.device_name = device.name.decode()
            info.device_memory_gb = total_mem / 1e9
            info.free_memory_gb = free_mem / 1e9
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")

    return info


if __name__ == "__main__":
    # Test the GPU detection
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing compute backend detection...\n")
    backend_type, backend_module = detect_compute_backend()

    print(f"\nBackend Type: {backend_type}")
    print(f"Module: {backend_module.__name__}")

    info = get_backend_info(backend_type, backend_module)
    print("\nDetailed Backend Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
