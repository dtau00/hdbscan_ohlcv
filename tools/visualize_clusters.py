#!/usr/bin/env python3
"""HDBSCAN OHLCV Cluster Visualization Tool

This tool visualizes sample OHLCV patterns from clusters discovered by HDBSCAN.
It creates candlestick plots arranged in a grid to show representative patterns
from each cluster.

Features:
- Parallel pattern rendering for faster visualization
- Representative sampling to avoid temporally overlapping patterns
- Customizable number of samples per cluster
- Flexible output formats

Usage:
    # Basic usage with parallel rendering
    python tools/visualize_clusters.py --run_id 1 --clusters 0 1 2 --n_samples 5

    # Visualize all clusters with parallel processing
    python tools/visualize_clusters.py --run_id run0001 --clusters all --n_samples 3

    # Custom output and specific number of parallel jobs
    python tools/visualize_clusters.py --run_id 2 --clusters 0 1 --output my_clusters.png --n-jobs 4

    # Disable parallel processing (use sequential)
    python tools/visualize_clusters.py --run_id 1 --clusters 0 1 --no-parallel
"""

import argparse
import logging
import sys
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import ResultsStorage
from src.data_loader import OHLCVDataLoader
from src.config import Config

logger = logging.getLogger(__name__)


def load_from_cache(cache_dir: Path, cluster_ids: Optional[List[int]] = None) -> Optional[Tuple[Dict[int, np.ndarray], Dict]]:
    """
    Load pre-generated cluster images from cache.

    Args:
        cache_dir: Directory containing cached images
        cluster_ids: Optional list of cluster IDs to load (None = load all)

    Returns:
        Tuple of (images_dict, metadata) if cache exists and is valid, None otherwise
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        logger.info(f"Cache directory does not exist: {cache_dir}")
        return None

    # Load metadata
    metadata_path = cache_dir / "metadata.pkl"
    if not metadata_path.exists():
        logger.warning(f"Cache metadata not found: {metadata_path}")
        return None

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"Found cache with {metadata['n_clusters']} clusters generated at {metadata['generated_at']}")

    # Determine which clusters to load
    if cluster_ids is None:
        cluster_ids_to_load = metadata['cluster_ids']
    else:
        cluster_ids_to_load = cluster_ids
        # Validate requested cluster IDs are in cache
        missing_ids = set(cluster_ids_to_load) - set(metadata['cluster_ids'])
        if missing_ids:
            logger.warning(f"Cluster IDs not in cache: {missing_ids}")
            return None

    # Load images
    images = {}
    for cluster_id in cluster_ids_to_load:
        img_path = cache_dir / f"cluster_{cluster_id}.png"
        if not img_path.exists():
            logger.warning(f"Missing image for cluster {cluster_id}: {img_path}")
            return None

        img = Image.open(img_path)
        images[cluster_id] = np.array(img)

    logger.info(f"Loaded {len(images)} cluster images from cache")
    return images, metadata


def _render_single_pattern(
    window: npt.NDArray[np.float64],
    title: str,
    show_grid: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Render a single OHLCV pattern to a numpy array.

    This is a standalone function for parallel processing.

    Args:
        window: OHLCV window of shape (window_size, 4)
        title: Title for the subplot
        show_grid: Whether to show grid lines

    Returns:
        Tuple of (rendered image as numpy array, title)
    """
    # Create a small figure for this pattern
    fig, ax = plt.subplots(figsize=(3, 2.5))

    window_size = len(window)

    # Extract OHLC data
    opens = window[:, 0]
    highs = window[:, 1]
    lows = window[:, 2]
    closes = window[:, 3]

    # Define colors
    color_up = '#26A69A'  # Green for up candles
    color_down = '#EF5350'  # Red for down candles

    # Plot each candlestick
    for i in range(window_size):
        open_price = opens[i]
        close_price = closes[i]
        high_price = highs[i]
        low_price = lows[i]

        # Determine color
        color = color_up if close_price >= open_price else color_down

        # Draw high-low line (wick)
        ax.plot([i, i], [low_price, high_price], color=color, linewidth=1.0)

        # Draw open-close box (body)
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)

        # Avoid zero height rectangles
        if height == 0:
            height = (high_price - low_price) * 0.01  # 1% of range

        rect = Rectangle(
            xy=(i - 0.3, bottom),
            width=0.6,
            height=height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0
        )
        ax.add_patch(rect)

    # Formatting
    ax.set_xlim(-0.5, window_size - 0.5)
    ax.set_ylim(window.min() * 0.995, window.max() * 1.005)
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlabel('Bar', fontsize=8)
    ax.set_ylabel('Price', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)

    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Convert to numpy array using modern matplotlib API
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB by dropping alpha channel
    img_array = img_array[:, :, :3]

    plt.close(fig)

    return img_array, title


def _render_cluster_image(
    cluster_id: int,
    cluster_info: Dict,
    windows: npt.NDArray[np.float64],
    sampled_indices: npt.NDArray[np.int64],
    n_samples: int,
    show_grid: bool = True
) -> Tuple[int, np.ndarray]:
    """
    Render a complete image for a single cluster with all its samples.

    This function creates a standalone figure with one row showing all samples
    from this cluster, then converts it to a numpy array image.

    Args:
        cluster_id: Cluster identifier
        cluster_info: Cluster information dict
        windows: All OHLCV windows
        sampled_indices: Sampled window indices for this cluster
        n_samples: Number of samples to render
        show_grid: Whether to show grid lines

    Returns:
        Tuple of (cluster_id, complete_row_image_array)
    """
    import os
    import threading
    import time

    start_time = time.time()
    pid = os.getpid()
    thread_name = threading.current_thread().name

    # Setup logging to file
    debug_log = Path("/tmp/hdbscan_viz_debug.log")
    with open(debug_log, 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] START Cluster {cluster_id} | PID={pid} | Thread={thread_name}\n")

    logger.info(f"[Cluster {cluster_id}] START rendering in PID={pid}, Thread={thread_name}")

    # Create figure for this cluster (1 row, n_samples columns)
    n_cols = min(len(sampled_indices), n_samples)

    with open(debug_log, 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] Cluster {cluster_id} | Creating figure with {n_cols} columns\n")
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3, 2.5), squeeze=False)

    # Plot each sample in this cluster
    for col_idx, window_idx in enumerate(sampled_indices):
        if col_idx >= n_cols:
            break

        ax = axes[0, col_idx]
        window = windows[window_idx]

        # Extract OHLC data
        window_size = len(window)
        opens = window[:, 0]
        highs = window[:, 1]
        lows = window[:, 2]
        closes = window[:, 3]

        # Define colors
        color_up = '#26A69A'
        color_down = '#EF5350'

        # Plot each candlestick
        for i in range(window_size):
            open_price = opens[i]
            close_price = closes[i]
            high_price = highs[i]
            low_price = lows[i]

            color = color_up if close_price >= open_price else color_down
            ax.plot([i, i], [low_price, high_price], color=color, linewidth=1.0)

            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            if height == 0:
                height = (high_price - low_price) * 0.01

            rect = Rectangle(
                xy=(i - 0.3, bottom),
                width=0.6,
                height=height,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0
            )
            ax.add_patch(rect)

        # Formatting
        ax.set_xlim(-0.5, window_size - 0.5)
        ax.set_ylim(window.min() * 0.995, window.max() * 1.005)

        # Create title
        if col_idx == 0:
            title = f"Cluster {cluster_id} (n={cluster_info['size']})\nSample {col_idx+1}"
        else:
            title = f"Sample {col_idx+1}"

        ax.set_title(title, fontsize=9, pad=5)
        ax.set_xlabel('Bar', fontsize=8)
        ax.set_ylabel('Price', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    with open(debug_log, 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] Cluster {cluster_id} | Plotting complete, converting to image\n")

    plt.tight_layout()

    # Convert entire figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB by dropping alpha channel
    img_array = img_array[:, :, :3]

    plt.close(fig)

    elapsed = time.time() - start_time
    with open(debug_log, 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] DONE Cluster {cluster_id} | Shape={img_array.shape} | Time={elapsed:.2f}s\n")

    logger.info(f"[Cluster {cluster_id}] DONE in {elapsed:.2f}s, shape={img_array.shape}")
    return cluster_id, img_array


class ClusterVisualizer:
    """
    Visualize OHLCV patterns from HDBSCAN clusters.

    This class loads clustering results and the original OHLCV data,
    then creates visualizations showing representative patterns from
    each cluster.

    Attributes:
        storage (ResultsStorage): Storage manager for loading results
        data_loader (Optional[OHLCVDataLoader]): Data loader for OHLCV data
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the cluster visualizer.

        Args:
            results_dir: Base directory for results (default: "results")
        """
        self.storage = ResultsStorage(results_dir)
        self.data_loader: Optional[OHLCVDataLoader] = None
        logger.info(f"Initialized ClusterVisualizer with results_dir: {results_dir}")

    def load_ohlcv_data(self, ohlcv_df: pd.DataFrame) -> None:
        """
        Load OHLCV data for visualization.

        Args:
            ohlcv_df: DataFrame with OHLCV columns
        """
        self.data_loader = OHLCVDataLoader(ohlcv_df, copy=False)
        logger.info(f"Loaded OHLCV data: {len(ohlcv_df)} bars")

    def load_run_data(
        self,
        run_id: int,
        ohlcv_df: pd.DataFrame
    ) -> Tuple[npt.NDArray[np.int32], Dict, npt.NDArray[np.float64]]:
        """
        Load all necessary data for a clustering run.

        Args:
            run_id: Run identifier
            ohlcv_df: OHLCV DataFrame to create windows from

        Returns:
            tuple: (labels, config, windows)
        """
        # Load labels and config
        labels, config = self.storage.load_labels(run_id)
        logger.info(f"Loaded labels for run {run_id}: {len(labels)} samples")

        # Load/create windows
        self.load_ohlcv_data(ohlcv_df)
        if self.data_loader is None:
            raise RuntimeError("Data loader not initialized")

        window_size = config['window_size']
        # Always create windows without batch_size to get the full array
        windows_result = self.data_loader.create_windows(window_size)

        # Ensure we have an array, not a generator
        if not isinstance(windows_result, np.ndarray):
            raise TypeError(f"Expected ndarray from create_windows, got {type(windows_result)}")

        windows: npt.NDArray[np.float64] = windows_result

        logger.info(f"Created windows: shape={windows.shape}")

        # Verify dimensions match
        if len(labels) != len(windows):
            raise ValueError(
                f"Mismatch between labels ({len(labels)}) and windows ({len(windows)}). "
                f"This suggests different data was used for clustering."
            )

        return labels, config, windows

    def get_cluster_info(
        self,
        labels: npt.NDArray[np.int32]
    ) -> Dict[int, Dict]:
        """
        Get information about each cluster.

        Args:
            labels: Cluster labels array

        Returns:
            Dictionary mapping cluster_id to info dict with:
                - size: number of samples
                - indices: array of sample indices
        """
        unique_labels = set(labels)

        cluster_info = {}
        for cluster_id in sorted(unique_labels):
            if cluster_id == -1:
                continue  # Skip noise

            indices = np.where(labels == cluster_id)[0]
            cluster_info[cluster_id] = {
                'size': len(indices),
                'indices': indices
            }

        return cluster_info

    def sample_cluster(
        self,
        cluster_indices: npt.NDArray[np.int64],
        n_samples: int,
        seed: Optional[int] = 42
    ) -> npt.NDArray[np.int64]:
        """
        Sample indices from a cluster.

        Args:
            cluster_indices: Array of indices belonging to cluster
            n_samples: Number of samples to draw
            seed: Random seed for reproducibility

        Returns:
            Array of sampled indices
        """
        if seed is not None:
            np.random.seed(seed)

        # If cluster is smaller than n_samples, return all indices
        if len(cluster_indices) <= n_samples:
            return cluster_indices

        # Random sample without replacement
        sampled = np.random.choice(
            cluster_indices,
            size=n_samples,
            replace=False
        )

        return sampled

    def sample_cluster_representatives(
        self,
        cluster_indices: npt.NDArray[np.int64],
        windows: npt.NDArray[np.float64],
        n_samples: int,
        min_distance: Optional[int] = None,
        window_size: Optional[int] = None,
        seed: Optional[int] = 42
    ) -> npt.NDArray[np.int64]:
        """
        Sample representative (non-overlapping) indices from a cluster.

        This method addresses the problem of temporally adjacent windows
        being clustered together. It selects diverse samples by:
        1. Computing distances to cluster centroid
        2. Selecting samples that are well-separated in time
        3. Prioritizing samples closer to the centroid

        Args:
            cluster_indices: Array of indices belonging to cluster
            windows: All OHLCV windows array
            n_samples: Number of samples to draw
            min_distance: Minimum temporal distance between samples (in window indices)
                         If None, defaults to window_size (no overlap)
            window_size: Size of windows (for default min_distance calculation)
                        If None, inferred from windows shape
            seed: Random seed for reproducibility

        Returns:
            Array of sampled indices that are temporally separated

        Example:
            >>> # Get 5 representative samples with minimum 10-bar separation
            >>> sampled = visualizer.sample_cluster_representatives(
            ...     cluster_indices, windows, n_samples=5, min_distance=10
            ... )
        """
        if seed is not None:
            np.random.seed(seed)

        # If cluster is smaller than n_samples, return all indices
        if len(cluster_indices) <= n_samples:
            return cluster_indices

        # Determine window size if not provided
        if window_size is None:
            window_size = windows.shape[1]

        # Set default minimum distance to window_size (no overlap)
        if min_distance is None:
            min_distance = window_size

        # Get cluster windows
        cluster_windows = windows[cluster_indices]

        # Compute centroid of cluster
        centroid = cluster_windows.mean(axis=0)

        # Compute distances to centroid for all cluster members
        # Flatten windows for distance calculation
        cluster_windows_flat = cluster_windows.reshape(len(cluster_windows), -1)
        centroid_flat = centroid.reshape(1, -1)

        distances = np.sqrt(np.sum((cluster_windows_flat - centroid_flat) ** 2, axis=1))

        # Create array of (index, distance) pairs, sorted by distance
        index_distance_pairs = list(zip(cluster_indices, distances))
        index_distance_pairs.sort(key=lambda x: x[1])  # Sort by distance to centroid

        # Greedy selection: pick samples that are far enough apart
        selected_indices = []
        for idx, dist in index_distance_pairs:
            # Check if this index is far enough from all already selected indices
            if not selected_indices or all(abs(idx - sel_idx) >= min_distance for sel_idx in selected_indices):
                selected_indices.append(idx)

                # Stop if we have enough samples
                if len(selected_indices) >= n_samples:
                    break

        # If we couldn't find enough separated samples, fill with closest remaining
        if len(selected_indices) < n_samples:
            logger.warning(
                f"Could only find {len(selected_indices)} separated samples "
                f"(requested {n_samples}) with min_distance={min_distance}"
            )
            # Add remaining closest samples that haven't been selected
            for idx, dist in index_distance_pairs:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= n_samples:
                        break

        return np.array(sorted(selected_indices), dtype=np.int64)

    def plot_ohlcv_window(
        self,
        ax: Axes,
        window: npt.NDArray[np.float64],
        title: str = "",
        show_grid: bool = True
    ) -> None:
        """
        Plot a single OHLCV window as a candlestick chart.

        Args:
            ax: Matplotlib axis to plot on
            window: OHLCV window of shape (window_size, 4)
            title: Title for the subplot
            show_grid: Whether to show grid lines
        """
        window_size = len(window)

        # Extract OHLC data
        opens = window[:, 0]
        highs = window[:, 1]
        lows = window[:, 2]
        closes = window[:, 3]

        # Define colors
        color_up = '#26A69A'  # Green for up candles
        color_down = '#EF5350'  # Red for down candles

        # Plot each candlestick
        for i in range(window_size):
            open_price = opens[i]
            close_price = closes[i]
            high_price = highs[i]
            low_price = lows[i]

            # Determine color
            color = color_up if close_price >= open_price else color_down

            # Draw high-low line (wick)
            ax.plot([i, i], [low_price, high_price], color=color, linewidth=1.0)

            # Draw open-close box (body)
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)

            # Avoid zero height rectangles
            if height == 0:
                height = (high_price - low_price) * 0.01  # 1% of range

            rect = Rectangle(
                xy=(i - 0.3, bottom),
                width=0.6,
                height=height,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0
            )
            ax.add_patch(rect)

        # Formatting
        ax.set_xlim(-0.5, window_size - 0.5)
        ax.set_ylim(window.min() * 0.995, window.max() * 1.005)
        ax.set_title(title, fontsize=9, pad=5)
        ax.set_xlabel('Bar', fontsize=8)
        ax.set_ylabel('Price', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    def plot_cluster_samples(
        self,
        run_id: int,
        ohlcv_df: pd.DataFrame,
        cluster_ids: Optional[List[int]] = None,
        n_samples: int = 5,
        output_path: Optional[str] = None,
        show: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        seed: int = 42,
        use_representatives: bool = True,
        min_distance: Optional[int] = None,
        use_parallel: bool = True,
        n_jobs: Optional[int] = None
    ) -> Optional[Path]:
        """
        Plot sample OHLCV windows from specified clusters.

        Args:
            run_id: Run identifier
            ohlcv_df: OHLCV DataFrame (same data used for clustering)
            cluster_ids: List of cluster IDs to visualize (None = all clusters)
            n_samples: Number of samples per cluster
            output_path: Path to save figure (if None, auto-generated)
            show: Whether to display the plot interactively
            figsize: Figure size in inches (width, height), auto-calculated if None
            seed: Random seed for sampling
            use_representatives: If True, select temporally separated representative samples.
                                If False, use random sampling.
            min_distance: Minimum temporal distance between samples (window indices).
                         Only used if use_representatives=True.
                         If None, defaults to window_size (no temporal overlap).
            use_parallel: If True, render patterns in parallel (default: True)
            n_jobs: Number of parallel jobs (None = auto, -1 = all cores)

        Returns:
            Path to saved figure, or None if not saved
        """
        # Load data
        logger.info(f"Loading data for run {run_id}...")
        labels, config, windows = self.load_run_data(run_id, ohlcv_df)

        # Get cluster info
        cluster_info = self.get_cluster_info(labels)

        if not cluster_info:
            logger.warning("No clusters found (all noise)")
            print("No clusters found - all points classified as noise")
            return None

        # Determine which clusters to plot
        if cluster_ids is None:
            cluster_ids = sorted(cluster_info.keys())
        else:
            # Validate cluster IDs
            invalid_ids = set(cluster_ids) - set(cluster_info.keys())
            if invalid_ids:
                raise ValueError(
                    f"Invalid cluster IDs: {invalid_ids}. "
                    f"Available clusters: {sorted(cluster_info.keys())}"
                )

        sampling_mode = "representative" if use_representatives else "random"
        execution_mode = "parallel" if use_parallel else "sequential"
        logger.info(
            f"Plotting {len(cluster_ids)} clusters with {n_samples} samples each "
            f"(sampling mode: {sampling_mode}, execution: {execution_mode})"
        )

        # Calculate grid dimensions
        n_clusters = len(cluster_ids)
        n_cols = n_samples  # Use actual n_samples requested
        n_rows = n_clusters

        # Calculate figure size if not provided
        if figsize is None:
            fig_width = n_cols * 3 + 1
            fig_height = n_rows * 2.5 + 1
            figsize = (int(fig_width), int(fig_height))

        # Create figure and axes
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            squeeze=False
        )

        # Get window size for representative sampling
        window_size = config['window_size']

        # Sample windows for all clusters first
        cluster_samples = []
        for cluster_id in cluster_ids:
            info = cluster_info[cluster_id]

            # Sample windows from this cluster
            if use_representatives:
                sampled_indices = self.sample_cluster_representatives(
                    info['indices'],
                    windows,
                    n_samples,
                    min_distance=min_distance,
                    window_size=window_size,
                    seed=seed
                )
            else:
                sampled_indices = self.sample_cluster(
                    info['indices'],
                    n_samples,
                    seed=seed
                )

            cluster_samples.append((cluster_id, info, sampled_indices))

        # Plot each cluster (sequential or parallel)
        if use_parallel:
            # Clear debug log at start
            debug_log = Path("/tmp/hdbscan_viz_debug.log")
            with open(debug_log, 'w') as f:
                f.write(f"=== Visualization Debug Log Started ===\n")
                f.write(f"Time: {pd.Timestamp.now()}\n")
                f.write(f"Clusters: {len(cluster_ids)}\n")
                f.write(f"Samples per cluster: {n_cols}\n")
                f.write("=" * 50 + "\n")

            # Determine number of jobs
            if n_jobs is None:
                n_jobs_actual = min(multiprocessing.cpu_count(), len(cluster_ids))
            elif n_jobs == -1:
                n_jobs_actual = multiprocessing.cpu_count()
            else:
                n_jobs_actual = min(n_jobs, len(cluster_ids))

            logger.info(f"Using {n_jobs_actual} parallel workers to render {len(cluster_ids)} cluster images")
            logger.info(f"Debug log: {debug_log}")

            # Render each cluster as a complete image in parallel
            # Use Python's native multiprocessing for true parallel execution
            try:
                logger.info(f"Starting parallel rendering with multiprocessing.Pool...")
                from multiprocessing import Pool
                from functools import partial

                # Create partial function with fixed parameters
                render_func = partial(
                    _render_cluster_image,
                    windows=windows,
                    n_samples=n_cols,
                    show_grid=True
                )

                # Prepare arguments
                args_list = [
                    (cluster_id, info, sampled_indices)
                    for cluster_id, info, sampled_indices in cluster_samples
                ]

                # Use multiprocessing Pool
                with Pool(processes=n_jobs_actual) as pool:
                    cluster_images = pool.starmap(
                        lambda cid, info, sidx: _render_cluster_image(cid, info, windows, sidx, n_cols, True),
                        args_list
                    )

                logger.info(f"Parallel rendering completed successfully with {len(cluster_images)} images")
            except Exception as e:
                logger.warning(f"Multiprocessing failed ({str(e)}), falling back to sequential mode...")
                cluster_images = []
                for cluster_id, info, sampled_indices in cluster_samples:
                    cluster_images.append(
                        _render_cluster_image(
                            cluster_id, info, windows, sampled_indices, n_cols
                        )
                    )

            # Place cluster images on axes (each cluster image spans entire row)
            for row_idx, (cluster_id, cluster_img) in enumerate(cluster_images):
                # Display the complete cluster image across the entire row
                # Turn off all individual axes and use the first one to show the full image
                for col_idx in range(n_cols):
                    axes[row_idx, col_idx].axis('off')

                # Use the entire row to display the cluster image
                axes[row_idx, 0].imshow(cluster_img, aspect='auto', extent=[0, n_cols, 0, 1])
                axes[row_idx, 0].set_xlim(0, n_cols)
                axes[row_idx, 0].set_ylim(0, 1)

        else:
            # Sequential rendering (original method)
            for row_idx, (cluster_id, info, sampled_indices) in enumerate(cluster_samples):
                # Plot each sample
                for col_idx, window_idx in enumerate(sampled_indices):
                    if col_idx >= n_cols:
                        break

                    ax = axes[row_idx, col_idx]
                    window = windows[window_idx]

                    # Create title
                    if col_idx == 0:
                        title = f"Cluster {cluster_id} (n={info['size']})\nSample {col_idx+1}"
                    else:
                        title = f"Sample {col_idx+1}"

                    self.plot_ohlcv_window(ax, window, title=title)

        # Hide unused subplots
        for row_idx, (cluster_id, info, sampled_indices) in enumerate(cluster_samples):
            for col_idx in range(len(sampled_indices), n_cols):
                axes[row_idx, col_idx].axis('off')

        # Add overall title
        config_id = Config.get_config_id(config)
        fig.suptitle(
            f'HDBSCAN Cluster Patterns - Run {run_id}\n'
            f'Config: {config_id} | {n_clusters} clusters',
            fontsize=14,
            y=0.995
        )

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))

        # Save figure
        output_file: Optional[Path] = None
        if output_path or not show:
            if output_path is None:
                # Use visualizations subdirectory under results
                vis_dir = Config.RESULTS_DIR / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                output_file = vis_dir / f"clusters_run{run_id:04d}_{config_id}.png"
            else:
                output_file = Path(output_path)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to: {output_file}")

        # Show interactively if requested
        if show:
            logger.info("Displaying plot...")
            plt.show()
        else:
            plt.close(fig)

        return output_file

    def print_cluster_summary(
        self,
        run_id: int,
        ohlcv_df: pd.DataFrame
    ) -> None:
        """
        Print summary statistics for clusters.

        Args:
            run_id: Run identifier
            ohlcv_df: OHLCV DataFrame
        """
        labels, config, _ = self.load_run_data(run_id, ohlcv_df)
        cluster_info = self.get_cluster_info(labels)

        n_noise = np.sum(labels == -1)
        n_total = len(labels)

        print("\n" + "=" * 70)
        print(f"Cluster Summary - Run {run_id}")
        print("=" * 70)

        print(f"\nConfiguration: {Config.get_config_id(config)}")
        print(f"  Window size: {config['window_size']}")
        print(f"  Min cluster size: {config['min_cluster_size']}")
        print(f"  Min samples: {config['min_samples']}")

        print(f"\nTotal samples: {n_total}")
        print(f"Total clusters: {len(cluster_info)}")
        print(f"Noise points: {n_noise} ({n_noise/n_total*100:.1f}%)")

        if cluster_info:
            print("\nCluster breakdown:")
            for cluster_id in sorted(cluster_info.keys()):
                info = cluster_info[cluster_id]
                print(f"  Cluster {cluster_id}: {info['size']} samples ({info['size']/n_total*100:.1f}%)")

        print("=" * 70 + "\n")


def main():
    """Main entry point for the cluster visualization tool."""
    parser = argparse.ArgumentParser(
        description='Visualize OHLCV patterns from HDBSCAN clusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize clusters 0, 1, 2 from run 1 with 5 representative samples each
  python tools/visualize_clusters.py --run_id 1 --clusters 0 1 2 --n_samples 5

  # Use random sampling instead of representative sampling
  python tools/visualize_clusters.py --run_id 1 --random_sampling

  # Set minimum temporal distance between samples to 15 bars
  python tools/visualize_clusters.py --run_id 1 --min_distance 15

  # Visualize all clusters from run 1
  python tools/visualize_clusters.py --run_id 1 --clusters all

  # Save to custom location
  python tools/visualize_clusters.py --run_id 2 --clusters 0 1 --output my_plot.png

  # Show cluster summary only
  python tools/visualize_clusters.py --run_id 1 --summary_only

  # Use custom OHLCV data file
  python tools/visualize_clusters.py --run_id 1 --data_file data/my_ohlcv.csv
        """
    )

    parser.add_argument(
        '--run_id',
        type=int,
        required=True,
        help='Run identifier (e.g., 1, 2, 3, ...)'
    )
    parser.add_argument(
        '--clusters',
        type=str,
        nargs='+',
        default=['all'],
        help='Cluster IDs to visualize (space-separated) or "all" for all clusters'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5,
        help='Number of samples per cluster (default: 5)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for the plot (default: auto-generated)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot interactively (requires display)'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        help='Path to OHLCV CSV file (if not provided, uses generate_sample_ohlcv)'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        help='Figure size as "width,height" in inches (default: auto-calculated)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--use_representatives',
        action='store_true',
        default=True,
        help='Use representative (temporally separated) sampling (default: True)'
    )
    parser.add_argument(
        '--random_sampling',
        action='store_true',
        help='Use random sampling instead of representative sampling'
    )
    parser.add_argument(
        '--min_distance',
        type=int,
        help='Minimum temporal distance between samples in bars (default: window_size)'
    )
    parser.add_argument(
        '--summary_only',
        action='store_true',
        help='Print cluster summary only, do not create visualization'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel pattern rendering (use sequential rendering)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel jobs for pattern rendering (None = auto, -1 = all cores)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        visualizer = ClusterVisualizer(results_dir=args.results_dir)

        # Load OHLCV data
        if args.data_file:
            logger.info(f"Loading OHLCV data from {args.data_file}")
            ohlcv_df = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
        else:
            logger.warning("No data file specified, using sample data")
            from src.data_loader import generate_sample_ohlcv
            ohlcv_df = generate_sample_ohlcv(n_bars=1000, seed=42)

        logger.info(f"Loaded OHLCV data: {len(ohlcv_df)} bars")

        # Print summary
        visualizer.print_cluster_summary(args.run_id, ohlcv_df)

        # Exit if summary only
        if args.summary_only:
            return 0

        # Parse cluster IDs
        if args.clusters == ['all']:
            cluster_ids = None  # Will plot all clusters
        else:
            try:
                cluster_ids = [int(c) for c in args.clusters]
            except ValueError:
                parser.error("Cluster IDs must be integers or 'all'")

        # Parse figsize
        figsize = None
        if args.figsize:
            try:
                figsize = tuple(map(int, args.figsize.split(',')))
                if len(figsize) != 2:
                    raise ValueError
            except ValueError:
                parser.error("--figsize must be in format 'width,height' (e.g., '15,10')")

        # Determine sampling mode
        use_representatives = not args.random_sampling

        # Create visualization
        print("\nGenerating cluster visualizations...")
        if use_representatives:
            print("Using representative sampling (temporally separated)")
        else:
            print("Using random sampling")

        use_parallel = not args.no_parallel
        if use_parallel:
            print(f"Using parallel pattern rendering (n_jobs={args.n_jobs or 'auto'})")
        else:
            print("Using sequential pattern rendering")

        output_path = visualizer.plot_cluster_samples(
            run_id=args.run_id,
            ohlcv_df=ohlcv_df,
            cluster_ids=cluster_ids,
            n_samples=args.n_samples,
            output_path=args.output,
            show=args.show,
            figsize=figsize,
            seed=args.seed,
            use_representatives=use_representatives,
            min_distance=args.min_distance,
            use_parallel=use_parallel,
            n_jobs=args.n_jobs
        )

        if output_path:
            print(f"\n✓ Visualization saved to: {output_path}")

        if args.show:
            print("✓ Plot displayed")

        print("\n✓ Cluster visualization completed successfully!\n")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
