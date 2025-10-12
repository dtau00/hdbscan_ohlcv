#!/usr/bin/env python3
"""
Standalone script to pre-generate cluster visualizations in parallel.

This script runs outside Streamlit and uses true multiprocessing to generate
cluster images in parallel, then caches them to disk for fast loading in the UI.

Usage:
    python tools/pregenerate_viz.py --results path/to/results.pkl [--jobs N] [--samples 10]
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time
from multiprocessing import Pool, cpu_count
import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _render_cluster_image(
    args: Tuple
) -> Tuple[int, np.ndarray]:
    """Wrapper for rendering that unpacks arguments."""
    cluster_id, cluster_info, windows, sampled_indices, n_samples, show_grid = args
    return _render_cluster_image_impl(cluster_id, cluster_info, windows, sampled_indices, n_samples, show_grid)


def _render_cluster_image_impl(
    cluster_id: int,
    cluster_info: Dict,
    windows: npt.NDArray[np.float64],
    sampled_indices: npt.NDArray[np.int64],
    n_samples: int,
    show_grid: bool = True
) -> Tuple[int, np.ndarray]:
    """
    Render a complete image for one cluster with all its samples.

    Args:
        cluster_id: The cluster ID to render
        cluster_info: Dictionary with 'indices' and 'size' for this cluster
        windows: All normalized windows array
        sampled_indices: Indices of samples to show for this cluster
        n_samples: Number of samples to display
        show_grid: Whether to show grid lines

    Returns:
        Tuple of (cluster_id, image_array)
    """
    import os
    pid = os.getpid()

    # Create figure for this cluster
    n_cols = len(sampled_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(f'Cluster {cluster_id} (Size: {cluster_info["size"]})',
                 fontsize=14, fontweight='bold')

    # Plot each sample
    for col_idx, sample_idx in enumerate(sampled_indices):
        ax = axes[col_idx]
        window = windows[sample_idx]

        # Extract OHLC data
        window_len = len(window)
        opens = window[:, 0]
        highs = window[:, 1]
        lows = window[:, 2]
        closes = window[:, 3]

        # Define colors
        color_up = '#26A69A'  # Green for up candles
        color_down = '#EF5350'  # Red for down candles

        # Plot each candlestick
        for i in range(window_len):
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

            from matplotlib.patches import Rectangle
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
        ax.set_xlim(-0.5, window_len - 0.5)
        ax.set_ylim(window.min() * 0.995, window.max() * 1.005)
        ax.set_title(f'Sample {col_idx + 1}', fontsize=10, pad=5)
        ax.set_xlabel('Bar', fontsize=9)
        ax.set_ylabel('Price', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)

    # Convert RGBA to RGB
    img_array = img_array[:, :, :3].copy()

    plt.close(fig)

    return cluster_id, img_array


def load_results(results_path: Path) -> Tuple[Dict, npt.NDArray]:
    """Load clustering results and windows from pickle file."""
    logger.info(f"Loading results from {results_path}")

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # Load windows
    windows_path = results_path.parent / 'windows.npy'
    if not windows_path.exists():
        raise FileNotFoundError(f"Windows file not found: {windows_path}")

    windows = np.load(windows_path)
    logger.info(f"Loaded {len(windows)} windows")

    return results, windows


def get_cluster_info(results: Dict) -> Dict[int, Dict]:
    """Extract cluster information from results."""
    labels = results.get('labels')
    if labels is None:
        raise ValueError("No labels found in results")

    cluster_info = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:  # Skip noise
            continue

        indices = np.where(labels == label)[0]
        cluster_info[int(label)] = {
            'indices': indices,
            'size': len(indices)
        }

    logger.info(f"Found {len(cluster_info)} clusters (excluding noise)")
    return cluster_info


def sample_cluster_indices(cluster_info: Dict[int, Dict], n_samples: int) -> Dict[int, npt.NDArray]:
    """Sample indices for each cluster."""
    sampled = {}

    for cluster_id, info in cluster_info.items():
        indices = info['indices']
        n_available = len(indices)

        if n_available <= n_samples:
            sampled[cluster_id] = indices
        else:
            # Sample evenly across the cluster
            step = n_available / n_samples
            sampled_idx = [int(i * step) for i in range(n_samples)]
            sampled[cluster_id] = indices[sampled_idx]

    return sampled


def generate_images_parallel(
    cluster_info: Dict[int, Dict],
    windows: npt.NDArray,
    sampled_indices: Dict[int, npt.NDArray],
    n_samples: int,
    n_jobs: int
) -> Dict[int, np.ndarray]:
    """Generate all cluster images in parallel using chunked approach."""

    cluster_ids = sorted(cluster_info.keys())

    logger.info(f"Starting parallel generation with {n_jobs} workers for {len(cluster_ids)} clusters")
    logger.info(f"Windows array shape: {windows.shape}, size: {windows.nbytes / (1024**2):.1f} MB")

    start_time = time.time()
    images_dict = {}

    # Use chunksize to avoid pickling huge arrays repeatedly
    chunk_size = max(1, len(cluster_ids) // (n_jobs * 4))

    # Prepare arguments for parallel processing
    args_list = [
        (cid, cluster_info[cid], windows, sampled_indices[cid], n_samples, True)
        for cid in cluster_ids
    ]

    try:
        with Pool(processes=n_jobs) as pool:
            # Process in chunks with progress
            for i, result in enumerate(pool.imap(_render_cluster_image, args_list, chunksize=chunk_size)):
                cluster_id, img = result
                images_dict[cluster_id] = img
                if (i + 1) % 20 == 0:
                    logger.info(f"Progress: {i+1}/{len(cluster_ids)} clusters rendered")
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        logger.info("Falling back to sequential processing...")

        # Fallback to sequential
        for cid in cluster_ids:
            args = (cid, cluster_info[cid], windows, sampled_indices[cid], n_samples, True)
            cluster_id, img = _render_cluster_image(args)
            images_dict[cluster_id] = img
            if (len(images_dict)) % 20 == 0:
                logger.info(f"Progress: {len(images_dict)}/{len(cluster_ids)} clusters rendered")

    elapsed = time.time() - start_time
    logger.info(f"Generated {len(images_dict)} images in {elapsed:.2f}s ({elapsed/len(images_dict):.3f}s per image)")

    return images_dict


def save_images_to_cache(images: Dict[int, np.ndarray], cache_dir: Path, metadata: Dict):
    """Save generated images to cache directory."""

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(images)} images to {cache_dir}")

    # Save each image
    for cluster_id, img_array in images.items():
        img_path = cache_dir / f"cluster_{cluster_id}.png"
        img = Image.fromarray(img_array)
        img.save(img_path, optimize=True)

    # Save metadata
    metadata_path = cache_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    logger.info(f"Cache saved to {cache_dir}")
    logger.info(f"Metadata: {metadata}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate cluster visualizations in parallel"
    )
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to results pickle file'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=None,
        help=f'Number of parallel jobs (default: all {cpu_count()} cores)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples per cluster (default: 10)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory (default: .cache/viz/<results_dir_name>)'
    )

    args = parser.parse_args()

    # Resolve paths
    results_path = Path(args.results).resolve()
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    # Determine cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        # Use results directory name for cache
        results_dir_name = results_path.parent.name
        cache_dir = Path('.cache/viz') / results_dir_name

    cache_dir = cache_dir.resolve()

    # Determine number of jobs
    n_jobs = args.jobs if args.jobs else cpu_count()
    logger.info(f"Using {n_jobs} parallel workers")

    # Load data
    results, windows = load_results(results_path)

    # Get cluster information
    cluster_info = get_cluster_info(results)

    # Sample indices
    sampled_indices = sample_cluster_indices(cluster_info, args.samples)

    # Generate images in parallel
    images = generate_images_parallel(
        cluster_info,
        windows,
        sampled_indices,
        args.samples,
        n_jobs
    )

    # Prepare metadata
    metadata = {
        'results_path': str(results_path),
        'n_clusters': len(cluster_info),
        'n_samples': args.samples,
        'generated_at': pd.Timestamp.now().isoformat(),
        'n_jobs': n_jobs,
        'cluster_ids': sorted(cluster_info.keys()),
        'cluster_sizes': {cid: info['size'] for cid, info in cluster_info.items()}
    }

    # Save to cache
    save_images_to_cache(images, cache_dir, metadata)

    logger.info("✓ Pre-generation complete!")
    logger.info(f"  Cache location: {cache_dir}")
    logger.info(f"  Total images: {len(images)}")
    logger.info(f"  Cluster IDs: {min(images.keys())} to {max(images.keys())}")


if __name__ == '__main__':
    main()
