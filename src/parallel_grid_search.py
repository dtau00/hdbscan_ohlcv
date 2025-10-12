#!/usr/bin/env python3
"""Parallel Grid Search for HDBSCAN OHLCV Pattern Discovery

This module provides parallel execution of hyperparameter grid search
using joblib for efficient multi-core processing.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import multiprocessing


def get_optimal_n_jobs(n_jobs: Optional[int] = None) -> int:
    """
    Determine optimal number of parallel jobs.

    Args:
        n_jobs: Requested number of jobs. None = auto-detect, -1 = all cores

    Returns:
        Number of jobs to use
    """
    cpu_count = multiprocessing.cpu_count()

    if n_jobs is None:
        # Use all cores by default
        return cpu_count
    elif n_jobs == -1:
        # Use all cores
        return cpu_count
    elif n_jobs < 1:
        # Use all except abs(n_jobs) cores
        return max(1, cpu_count + n_jobs)
    else:
        # Use specified number of cores
        return min(n_jobs, cpu_count)


def process_config_wrapper(
    config: Dict[str, Any],
    df_ohlcv: pd.DataFrame,
    backend_type: str,
    backend_module: Any,
    storage: Any,
    scalers_cache: Dict[int, StandardScaler],
    process_func: Callable,
    file_display: Optional[str] = None
) -> Dict[str, Any]:
    """
    Wrapper function for parallel processing of a single configuration.

    This wrapper is needed because joblib requires top-level functions
    for pickling in multiprocessing mode.

    Args:
        config: Configuration dictionary
        df_ohlcv: OHLCV DataFrame
        backend_type: Compute backend type
        backend_module: Backend module reference (or None for CPU)
        storage: ResultsStorage instance
        scalers_cache: Cache of fitted scalers
        process_func: Function to process the config
        file_display: Optional file name for display

    Returns:
        Result dictionary
    """
    # Create a local logger for this process
    logger = logging.getLogger(f"parallel_worker_{multiprocessing.current_process().name}")

    # Log which configuration is being processed
    from src.config import Config
    config_id = Config.get_config_id(config)
    file_info = f" on {file_display}" if file_display else ""
    logger.info(f"[PARALLEL] Starting config {config_id}{file_info} (PID: {multiprocessing.current_process().pid})")

    # If backend_module is None and we're using CPU, import hdbscan
    actual_backend_module = backend_module
    if backend_type == 'cpu' and actual_backend_module is None:
        try:
            import hdbscan
            actual_backend_module = hdbscan
        except ImportError:
            logger.error("hdbscan module not found for CPU backend")
            actual_backend_module = None

    result = process_func(
        config=config,
        df_ohlcv=df_ohlcv,
        backend_type=backend_type,
        backend_module=actual_backend_module,
        storage=storage,
        logger=logger,
        scalers_cache=scalers_cache
    )

    # Log completion
    success_status = "SUCCESS" if result.get('success', False) else "FAILED"
    logger.info(f"[PARALLEL] Completed config {config_id}{file_info} - {success_status} (PID: {multiprocessing.current_process().pid})")

    # Add file info if provided
    if file_display:
        result['data_file'] = file_display

    return result


def parallel_grid_search(
    configs: List[Dict[str, Any]],
    df_ohlcv: pd.DataFrame,
    backend_type: str,
    backend_module: Any,
    storage: Any,
    process_func: Callable,
    n_jobs: Optional[int] = None,
    verbose: int = 10,
    file_display: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute grid search in parallel using joblib.

    Note: When using GPU backend, parallel execution will force CPU-only mode
    for worker processes because CUDA contexts cannot be shared across forked processes.

    Args:
        configs: List of configuration dictionaries
        df_ohlcv: OHLCV DataFrame
        backend_type: Compute backend type
        backend_module: Backend module reference
        storage: ResultsStorage instance
        process_func: Function to process each config
        n_jobs: Number of parallel jobs (-1 = all cores, None = auto)
        verbose: Verbosity level for joblib (0-100)
        file_display: Optional file name for display

    Returns:
        List of result dictionaries
    """
    n_jobs_actual = get_optimal_n_jobs(n_jobs)

    # Force CPU backend for parallel execution
    # CUDA contexts cannot be shared across forked processes
    parallel_backend_type = 'cpu'
    parallel_backend_module = None

    # Create shared scaler cache (one per window size)
    # Note: In parallel execution, each process will maintain its own cache
    scalers_cache = {}

    # Execute in parallel
    results = Parallel(n_jobs=n_jobs_actual, verbose=verbose, backend='loky')(
        delayed(process_config_wrapper)(
            config=config,
            df_ohlcv=df_ohlcv,
            backend_type=parallel_backend_type,
            backend_module=parallel_backend_module,
            storage=storage,
            scalers_cache=scalers_cache,
            process_func=process_func,
            file_display=file_display
        )
        for config in configs
    )

    return results


def parallel_multi_file_grid_search(
    configs: List[Dict[str, Any]],
    data_files: List[Any],
    backend_type: str,
    backend_module: Any,
    storage: Any,
    process_func: Callable,
    n_jobs: Optional[int] = None,
    verbose: int = 10
) -> List[Dict[str, Any]]:
    """
    Execute grid search in parallel across multiple data files.

    Creates a Cartesian product of configs × data_files and processes
    all combinations in parallel.

    Note: When using GPU backend, parallel execution will force CPU-only mode
    for worker processes because CUDA contexts cannot be shared across forked processes.

    Args:
        configs: List of configuration dictionaries
        data_files: List of (df_ohlcv, file_display) tuples
        backend_type: Compute backend type
        backend_module: Backend module reference
        storage: ResultsStorage instance
        process_func: Function to process each config
        n_jobs: Number of parallel jobs (-1 = all cores, None = auto)
        verbose: Verbosity level for joblib (0-100)

    Returns:
        List of result dictionaries
    """
    n_jobs_actual = get_optimal_n_jobs(n_jobs)

    # Force CPU backend for parallel execution
    # CUDA contexts cannot be shared across forked processes
    parallel_backend_type = 'cpu'
    parallel_backend_module = None

    # Create list of all (config, data) combinations
    tasks = []
    for df_ohlcv, file_display in data_files:
        for config in configs:
            tasks.append((config, df_ohlcv, file_display))

    # Create shared scaler cache
    scalers_cache = {}

    # Execute all combinations in parallel
    results = Parallel(n_jobs=n_jobs_actual, verbose=verbose, backend='loky')(
        delayed(process_config_wrapper)(
            config=config,
            df_ohlcv=df_ohlcv,
            backend_type=parallel_backend_type,
            backend_module=parallel_backend_module,
            storage=storage,
            scalers_cache=scalers_cache,
            process_func=process_func,
            file_display=file_display
        )
        for config, df_ohlcv, file_display in tasks
    )

    return results
