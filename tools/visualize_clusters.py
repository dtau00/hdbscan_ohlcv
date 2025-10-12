#!/usr/bin/env python3
"""HDBSCAN OHLCV Cluster Visualization Tool

This tool visualizes sample OHLCV patterns from clusters discovered by HDBSCAN.
It creates candlestick plots arranged in a grid to show representative patterns
from each cluster.

Usage:
    python tools/visualize_clusters.py --run_id 1 --clusters 0 1 2 --n_samples 5
    python tools/visualize_clusters.py --run_id run0001 --clusters all --n_samples 3
    python tools/visualize_clusters.py --run_id 2 --clusters 0 1 --output my_clusters.png
"""

import argparse
import logging
import sys
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import ResultsStorage
from src.data_loader import OHLCVDataLoader
from src.config import Config

logger = logging.getLogger(__name__)


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
        seed: int = 42
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

        logger.info(f"Plotting {len(cluster_ids)} clusters with {n_samples} samples each")

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

        # Plot each cluster
        for row_idx, cluster_id in enumerate(cluster_ids):
            info = cluster_info[cluster_id]

            # Sample windows from this cluster
            sampled_indices = self.sample_cluster(
                info['indices'],
                n_samples,
                seed=seed
            )

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

            # Hide unused subplots in this row
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
  # Visualize clusters 0, 1, 2 from run 1 with 5 samples each
  python tools/visualize_clusters.py --run_id 1 --clusters 0 1 2 --n_samples 5

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
        '--summary_only',
        action='store_true',
        help='Print cluster summary only, do not create visualization'
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

        # Create visualization
        print("\nGenerating cluster visualizations...")
        output_path = visualizer.plot_cluster_samples(
            run_id=args.run_id,
            ohlcv_df=ohlcv_df,
            cluster_ids=cluster_ids,
            n_samples=args.n_samples,
            output_path=args.output,
            show=args.show,
            figsize=figsize,
            seed=args.seed
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
