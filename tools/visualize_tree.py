#!/usr/bin/env python3
"""HDBSCAN Cluster Tree Visualization Tool

This tool loads a saved HDBSCAN clusterer object and visualizes its
condensed cluster tree to understand the hierarchical clustering structure.

Usage:
    python tools/visualize_tree.py --run_id run0001_ws10_mcs10_ms6_euclidean_eom
    python tools/visualize_tree.py --run_id run0002 --output tree_run0002.png
    python tools/visualize_tree.py --run_id run0001 --show
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, Any

import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ClusterTreeVisualizer:
    """
    Visualize HDBSCAN cluster trees from saved clusterer objects.

    This class loads pickled HDBSCAN clusterer objects and creates
    visualizations of their condensed trees, showing the hierarchical
    structure of the clustering.

    Attributes:
        results_dir (Path): Directory containing saved results
        models_dir (Path): Directory containing saved clusterer models
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the tree visualizer.

        Args:
            results_dir: Base directory for results (default: "results")
        """
        self.results_dir = Path(results_dir)
        self.models_dir = self.results_dir / "models"

        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir}\n"
                f"Make sure you have run main.py to generate clusterer models."
            )

        logger.info(f"Initialized ClusterTreeVisualizer with models_dir: {self.models_dir}")

    def load_clusterer(self, run_id: str) -> Tuple[Any, Path]:
        """
        Load a saved clusterer object by run ID.

        Args:
            run_id: Run identifier (e.g., "run0001_ws10_mcs10_ms6_euclidean_eom"
                    or just "run0001")

        Returns:
            tuple: (clusterer_object, file_path)

        Raises:
            FileNotFoundError: If clusterer file not found
            RuntimeError: If loading fails
        """
        # Find matching clusterer file
        pattern = f"clusterer_{run_id}*.pkl"
        matching_files = list(self.models_dir.glob(pattern))

        if not matching_files:
            # Try with just the run number
            pattern = f"clusterer_run{run_id.lstrip('run')}*.pkl"
            matching_files = list(self.models_dir.glob(pattern))

        if not matching_files:
            available = [f.stem for f in self.models_dir.glob("clusterer_*.pkl")]
            raise FileNotFoundError(
                f"No clusterer found matching '{run_id}'\n"
                f"Available clusterers:\n" + "\n".join(f"  - {a}" for a in available)
            )

        if len(matching_files) > 1:
            logger.warning(
                f"Multiple files match '{run_id}': {[f.name for f in matching_files]}\n"
                f"Using first match: {matching_files[0].name}"
            )

        clusterer_path = matching_files[0]
        logger.info(f"Loading clusterer from: {clusterer_path}")

        try:
            # Try joblib first (used by storage.py)
            clusterer = joblib.load(clusterer_path)
            logger.info("Clusterer loaded successfully (joblib)")
            return clusterer, clusterer_path
        except Exception as e_joblib:
            error_msg = str(e_joblib).lower()

            # Check if it's a CUDA-related error
            if 'cuda' in error_msg or 'gpu' in error_msg:
                raise RuntimeError(
                    f"Failed to load GPU-based clusterer from {clusterer_path}\n\n"
                    f"Error: {e_joblib}\n\n"
                    f"REASON: This clusterer was saved using cuML (GPU backend) and requires\n"
                    f"CUDA/GPU to load. The condensed tree is not available with cuML.\n\n"
                    f"SOLUTION: To visualize cluster trees, re-run the clustering with CPU backend:\n"
                    f"  1. Temporarily disable GPU in src/gpu_utils.py, or\n"
                    f"  2. Run on a system without CUDA, or\n"
                    f"  3. Set CUDA_VISIBLE_DEVICES=-1 to force CPU mode\n\n"
                    f"Note: CPU backend (hdbscan library) provides full tree visualization support."
                ) from e_joblib

            logger.warning(f"joblib load failed: {e_joblib}, trying pickle")
            try:
                with open(clusterer_path, 'rb') as f:
                    clusterer = pickle.load(f)
                logger.info("Clusterer loaded successfully (pickle)")
                return clusterer, clusterer_path
            except Exception as e_pickle:
                raise RuntimeError(
                    f"Failed to load clusterer from {clusterer_path}\n"
                    f"  joblib error: {e_joblib}\n"
                    f"  pickle error: {e_pickle}"
                ) from e_pickle

    def check_tree_availability(self, clusterer: Any) -> bool:
        """
        Check if the clusterer has a condensed tree available.

        Args:
            clusterer: Loaded clusterer object

        Returns:
            bool: True if condensed_tree_ is available
        """
        has_tree = hasattr(clusterer, 'condensed_tree_')

        if has_tree:
            logger.info("✓ Condensed tree is available")
        else:
            logger.warning(
                "✗ Condensed tree not available\n"
                "This may happen with GPU (cuML) backend clusterers.\n"
                "Try using CPU backend for tree visualization."
            )

        return has_tree

    def plot_tree(
        self,
        clusterer: Any,
        output_path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        **kwargs
    ) -> Optional[Path]:
        """
        Plot the condensed cluster tree.

        Args:
            clusterer: Loaded clusterer object with condensed_tree_
            output_path: Path to save figure (if None, auto-generated)
            show: Whether to display the plot interactively
            figsize: Figure size in inches (width, height)
            **kwargs: Additional arguments for tree plotting

        Returns:
            Path to saved figure, or None if not saved

        Raises:
            RuntimeError: If tree plotting fails
        """
        if not self.check_tree_availability(clusterer):
            raise RuntimeError(
                "Cannot plot tree: condensed_tree_ not available in clusterer"
            )

        try:
            logger.info("Creating condensed tree plot...")

            fig, ax = plt.subplots(figsize=figsize)

            # Plot the condensed tree
            # Note: This works with CPU backend (hdbscan library)
            clusterer.condensed_tree_.plot(
                axis=ax,
                select_clusters=True,
                selection_palette=plt.cm.viridis.colors,
                **kwargs
            )

            ax.set_title(
                'HDBSCAN Condensed Cluster Tree\n'
                'Shows hierarchical cluster formation',
                fontsize=14,
                pad=20
            )
            ax.set_xlabel('Samples (ordered by cluster membership)', fontsize=11)
            ax.set_ylabel('Lambda (1 / distance)', fontsize=11)

            plt.tight_layout()

            # Save figure
            if output_path or not show:
                if output_path is None:
                    output_path = self.results_dir / "visualizations" / "cluster_tree.png"
                else:
                    output_path = Path(output_path)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Tree plot saved to: {output_path}")

            # Show interactively if requested
            if show:
                logger.info("Displaying plot...")
                plt.show()
            else:
                plt.close(fig)

            return Path(output_path) if output_path else None

        except Exception as e:
            raise RuntimeError(f"Failed to plot cluster tree: {e}") from e

    def print_tree_stats(self, clusterer: Any) -> None:
        """
        Print statistics about the cluster tree.

        Args:
            clusterer: Loaded clusterer object
        """
        print("\n" + "=" * 70)
        print("Cluster Tree Statistics")
        print("=" * 70)

        # Basic clustering info
        if hasattr(clusterer, 'labels_'):
            labels = clusterer.labels_
            if hasattr(labels, 'get'):  # CuPy array
                import cupy as cp
                labels = cp.asnumpy(labels)

            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)
            n_samples = len(labels)

            print(f"Total samples: {n_samples}")
            print(f"Clusters found: {n_clusters}")
            print(f"Noise points: {n_noise} ({n_noise/n_samples*100:.1f}%)")
            print(f"Cluster labels: {sorted([l for l in unique_labels if l != -1])}")

            # Cluster sizes
            print("\nCluster sizes:")
            for cluster_id in sorted([l for l in unique_labels if l != -1]):
                size = np.sum(labels == cluster_id)
                print(f"  Cluster {cluster_id}: {size} samples ({size/n_samples*100:.1f}%)")

        # Tree-specific info
        if hasattr(clusterer, 'condensed_tree_'):
            tree = clusterer.condensed_tree_
            print(f"\nCondensed tree available: Yes")
            print(f"  Tree data shape: {tree._raw_tree.shape if hasattr(tree, '_raw_tree') else 'N/A'}")
        else:
            print(f"\nCondensed tree available: No")

        # Cluster persistence
        if hasattr(clusterer, 'cluster_persistence_'):
            persistence = clusterer.cluster_persistence_
            if hasattr(persistence, 'get'):  # CuPy array
                import cupy as cp
                persistence = cp.asnumpy(persistence)
            print(f"\nCluster persistence:")
            for i, p in enumerate(persistence):
                print(f"  Cluster {i}: {p:.4f}")

        # Probabilities
        if hasattr(clusterer, 'probabilities_'):
            probs = clusterer.probabilities_
            if hasattr(probs, 'get'):  # CuPy array
                import cupy as cp
                probs = cp.asnumpy(probs)
            print(f"\nCluster membership probabilities:")
            print(f"  Mean: {probs.mean():.3f}")
            print(f"  Std: {probs.std():.3f}")
            print(f"  Min: {probs.min():.3f}")
            print(f"  Max: {probs.max():.3f}")

        print("=" * 70 + "\n")


def main():
    """Main entry point for the tree visualization tool."""
    parser = argparse.ArgumentParser(
        description='Visualize HDBSCAN cluster tree from saved clusterer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize tree for run 0001
  python tools/visualize_tree.py --run_id run0001

  # Specify full run ID with parameters
  python tools/visualize_tree.py --run_id run0001_ws10_mcs10_ms6_euclidean_eom

  # Save to custom location
  python tools/visualize_tree.py --run_id run0002 --output my_tree.png

  # Show interactively (requires display)
  python tools/visualize_tree.py --run_id run0001 --show

  # List available runs
  python tools/visualize_tree.py --list
        """
    )

    parser.add_argument(
        '--run_id',
        type=str,
        help='Run identifier (e.g., "run0001" or "run0001_ws10_mcs10_ms6_euclidean_eom")'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for the tree plot (default: results/visualizations/cluster_tree.png)'
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
        '--list',
        action='store_true',
        help='List available clusterer models and exit'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        default='12,8',
        help='Figure size as "width,height" in inches (default: 12,8)'
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
        visualizer = ClusterTreeVisualizer(results_dir=args.results_dir)

        # List available models
        if args.list:
            print("\n" + "=" * 70)
            print("Available Clusterer Models")
            print("=" * 70)
            models = sorted(visualizer.models_dir.glob("clusterer_*.pkl"))
            if models:
                for model_path in models:
                    # Extract run ID from filename
                    run_id = model_path.stem.replace("clusterer_", "")
                    size_mb = model_path.stat().st_size / 1024 / 1024
                    print(f"  {run_id} ({size_mb:.2f} MB)")
                print(f"\nTotal: {len(models)} models")
            else:
                print("  No models found.")
            print("=" * 70 + "\n")
            return 0

        # Validate run_id
        if not args.run_id:
            parser.error("--run_id is required (or use --list to see available runs)")

        # Parse figsize
        try:
            figsize = tuple(map(int, args.figsize.split(',')))
            if len(figsize) != 2:
                raise ValueError
        except ValueError:
            parser.error("--figsize must be in format 'width,height' (e.g., '12,8')")

        # Load clusterer
        print("\n" + "=" * 70)
        print(f"Loading Clusterer: {args.run_id}")
        print("=" * 70)
        clusterer, clusterer_path = visualizer.load_clusterer(args.run_id)
        print(f"✓ Loaded from: {clusterer_path.name}\n")

        # Print statistics
        visualizer.print_tree_stats(clusterer)

        # Plot tree
        if visualizer.check_tree_availability(clusterer):
            print("Generating tree visualization...")
            output_path = visualizer.plot_tree(
                clusterer,
                output_path=args.output,
                show=args.show,
                figsize=figsize
            )

            if output_path:
                print(f"\n✓ Tree visualization saved to: {output_path}")

            if args.show:
                print("✓ Plot displayed")

            print("\n✓ Tree visualization completed successfully!\n")
        else:
            print("\n✗ Cannot create visualization: tree data not available")
            print("Tip: Re-run clustering with CPU backend to generate tree data\n")
            return 1

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
