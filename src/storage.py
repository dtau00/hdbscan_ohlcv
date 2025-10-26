"""Results Storage for HDBSCAN OHLCV Clustering

This module provides functionality to save and load clustering results,
including metrics, labels, and fitted clusterer objects.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np
import numpy.typing as npt
import pandas as pd
import joblib

from src.config import Config

logger = logging.getLogger(__name__)


class ResultsStorage:
    """
    Manages saving and loading of clustering results.

    Handles storage of:
    - Metrics CSV file with all run results
    - Cluster labels as compressed numpy arrays
    - Fitted clusterer objects for later analysis
    - Configuration metadata

    Attributes:
        output_dir (Path): Root directory for results
        metrics_file (Path): Path to metrics CSV file
        run_counter (int): Counter for unique run IDs
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the results storage manager.

        Args:
            output_dir: Root directory for results. If None, uses Config.RESULTS_DIR

        Examples:
            >>> storage = ResultsStorage()
            >>> # Uses default results directory

            >>> storage = ResultsStorage("/path/to/custom/results")
            >>> # Uses custom directory
        """
        if output_dir is None:
            self.output_dir = Config.RESULTS_DIR
        else:
            self.output_dir = Path(output_dir)

        # Create directory structure
        self._ensure_directories()

        # Set up file paths
        self.metrics_file = Config.METRICS_DIR / "metrics.csv"

        # Initialize run counter
        self.run_counter = self._get_next_run_id()

        logger.info(f"ResultsStorage initialized: {self.output_dir}")
        logger.info(f"Next run ID: {self.run_counter}")

    def _ensure_directories(self) -> None:
        """Create necessary directory structure."""
        Config.ensure_dirs()
        logger.debug("Ensured all result directories exist")

    def _get_next_run_id(self) -> int:
        """
        Get the next available run ID by checking existing files.

        Returns:
            Next available run ID (integer)
        """
        if not self.metrics_file.exists():
            return 1

        try:
            df = pd.read_csv(self.metrics_file)
            if 'run_id' in df.columns and len(df) > 0:
                return int(df['run_id'].max()) + 1
            else:
                return 1
        except Exception as e:
            logger.warning(f"Could not read metrics file to get run ID: {e}")
            return 1

    def save_run_results(
        self,
        config: Dict[str, Any],
        labels: npt.NDArray[np.int32],
        metrics: Dict[str, Any],
        clusterer: Optional[Any] = None,
        features: Optional[npt.NDArray[np.float64]] = None,
        save_clusterer: bool = True,
        save_features: bool = False
    ) -> int:
        """
        Save results for a single HDBSCAN run.

        Args:
            config: Configuration dictionary for this run
            labels: Cluster labels array
            metrics: Metrics dictionary from ClusterMetrics.compute_metrics()
            clusterer: Fitted HDBSCAN clusterer object (optional)
            features: Feature matrix used for clustering (optional)
            save_clusterer: Whether to save the clusterer object
            save_features: Whether to save the feature matrix

        Returns:
            run_id: Unique ID for this run

        Examples:
            >>> storage = ResultsStorage()
            >>> run_id = storage.save_run_results(
            ...     config={'window_size': 10, 'min_cluster_size': 5, 'min_samples': 6,
            ...             'metric': 'euclidean', 'cluster_selection_method': 'eom'},
            ...     labels=labels_array,
            ...     metrics=metrics_dict,
            ...     clusterer=fitted_clusterer
            ... )
            >>> print(f"Saved run ID: {run_id}")
        """
        run_id = self.run_counter
        self.run_counter += 1

        logger.info(f"Saving results for run {run_id}")

        # Create run record for metrics CSV
        run_record = self._create_run_record(run_id, config, metrics)

        # Save metrics to CSV
        self._append_metrics_to_csv(run_record)

        # Save labels
        labels_path = self._save_labels(run_id, labels, config)
        logger.debug(f"Saved labels to {labels_path}")

        # Save clusterer if requested
        if save_clusterer and clusterer is not None:
            clusterer_path = self._save_clusterer(run_id, clusterer, config)
            logger.debug(f"Saved clusterer to {clusterer_path}")

        # Save features if requested
        if save_features and features is not None:
            features_path = self._save_features(run_id, features, config)
            logger.debug(f"Saved features to {features_path}")

        logger.info(f"Run {run_id} saved successfully")
        return run_id

    def _create_run_record(
        self,
        run_id: int,
        config: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a record for the metrics CSV.

        Args:
            run_id: Unique run identifier
            config: Configuration dictionary
            metrics: Metrics dictionary

        Returns:
            Dictionary with all relevant information for CSV row
        """
        record = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'config_id': Config.get_config_id(config),

            # Configuration parameters
            'window_size': config['window_size'],
            'stride': config.get('stride', 1),  # Default to 1 if not present (backwards compatibility)
            'min_cluster_size': config['min_cluster_size'],
            'min_samples': config['min_samples'],
            'metric': config['metric'],
            'cluster_selection_method': config['cluster_selection_method'],

            # Basic metrics
            'n_clusters': metrics['n_clusters'],
            'n_noise_points': metrics['n_noise_points'],
            'noise_ratio': metrics['noise_ratio'],

            # Quality metrics
            'silhouette_score': metrics.get('silhouette_score'),
            'davies_bouldin_score': metrics.get('davies_bouldin_score'),
            'calinski_harabasz_score': metrics.get('calinski_harabasz_score'),
        }

        # Add cluster size statistics
        if metrics['cluster_sizes']:
            record['min_cluster_size_actual'] = min(metrics['cluster_sizes'])
            record['max_cluster_size_actual'] = max(metrics['cluster_sizes'])
            record['mean_cluster_size'] = np.mean(metrics['cluster_sizes'])
            record['median_cluster_size'] = np.median(metrics['cluster_sizes'])
        else:
            record['min_cluster_size_actual'] = None
            record['max_cluster_size_actual'] = None
            record['mean_cluster_size'] = None
            record['median_cluster_size'] = None

        return record

    def _append_metrics_to_csv(self, record: Dict[str, Any]) -> None:
        """
        Append a metrics record to the CSV file.

        Args:
            record: Dictionary with metrics data
        """
        # Create DataFrame from record
        df_new = pd.DataFrame([record])

        # Append to existing file or create new one
        if self.metrics_file.exists():
            df_new.to_csv(self.metrics_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.metrics_file, index=False)

        logger.debug(f"Appended metrics for run {record['run_id']} to {self.metrics_file}")

    def _save_labels(
        self,
        run_id: int,
        labels: npt.NDArray[np.int32],
        config: Dict[str, Any]
    ) -> Path:
        """
        Save cluster labels as compressed numpy array.

        Args:
            run_id: Run identifier
            labels: Cluster labels array
            config: Configuration dictionary

        Returns:
            Path to saved labels file
        """
        config_id = Config.get_config_id(config)
        filename = f"labels_run{run_id:04d}_{config_id}.npz"
        filepath = Config.LABELS_DIR / filename

        # Save with compression
        np.savez_compressed(filepath, labels=labels, config=config)

        return filepath

    def _save_clusterer(
        self,
        run_id: int,
        clusterer: Any,
        config: Dict[str, Any]
    ) -> Path:
        """
        Save fitted clusterer object using joblib.

        Args:
            run_id: Run identifier
            clusterer: Fitted HDBSCAN clusterer
            config: Configuration dictionary

        Returns:
            Path to saved clusterer file
        """
        config_id = Config.get_config_id(config)
        filename = f"clusterer_run{run_id:04d}_{config_id}.pkl"
        filepath = Config.MODELS_DIR / filename

        try:
            # Try joblib first (better for sklearn-like objects)
            joblib.dump(clusterer, filepath, compress=3)
        except Exception as e:
            logger.warning(f"joblib dump failed: {e}, trying pickle")
            # Fallback to pickle
            with open(filepath, 'wb') as f:
                pickle.dump(clusterer, f, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath

    def _save_features(
        self,
        run_id: int,
        features: npt.NDArray[np.float64],
        config: Dict[str, Any]
    ) -> Path:
        """
        Save feature matrix as compressed numpy array.

        Args:
            run_id: Run identifier
            features: Feature matrix
            config: Configuration dictionary

        Returns:
            Path to saved features file
        """
        config_id = Config.get_config_id(config)
        filename = f"features_run{run_id:04d}_{config_id}.npz"
        filepath = Config.LABELS_DIR / filename

        # Save with compression
        np.savez_compressed(filepath, features=features, config=config)

        return filepath

    def load_labels(self, run_id: int) -> tuple[npt.NDArray[np.int32], Dict[str, Any]]:
        """
        Load cluster labels for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            tuple: (labels array, config dictionary)

        Raises:
            FileNotFoundError: If labels file not found for run_id

        Examples:
            >>> storage = ResultsStorage()
            >>> labels, config = storage.load_labels(run_id=1)
            >>> print(f"Loaded {len(labels)} labels")
        """
        # Find the labels file for this run_id
        pattern = f"labels_run{run_id:04d}_*.npz"
        matches = list(Config.LABELS_DIR.glob(pattern))

        if not matches:
            raise FileNotFoundError(f"No labels file found for run_id {run_id}")

        if len(matches) > 1:
            logger.warning(f"Multiple labels files found for run_id {run_id}, using first one")

        filepath = matches[0]
        logger.debug(f"Loading labels from {filepath}")

        # Load from compressed numpy file
        data = np.load(filepath, allow_pickle=True)
        labels = data['labels']
        config = data['config'].item() if 'config' in data else {}

        return labels, config

    def load_clusterer(self, run_id: int) -> Any:
        """
        Load fitted clusterer object for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Fitted clusterer object

        Raises:
            FileNotFoundError: If clusterer file not found for run_id

        Examples:
            >>> storage = ResultsStorage()
            >>> clusterer = storage.load_clusterer(run_id=1)
            >>> # Can now access clusterer.condensed_tree_, etc.
        """
        # Find the clusterer file for this run_id
        pattern = f"clusterer_run{run_id:04d}_*.pkl"
        matches = list(Config.MODELS_DIR.glob(pattern))

        if not matches:
            raise FileNotFoundError(f"No clusterer file found for run_id {run_id}")

        if len(matches) > 1:
            logger.warning(f"Multiple clusterer files found for run_id {run_id}, using first one")

        filepath = matches[0]
        logger.debug(f"Loading clusterer from {filepath}")

        # Try loading with joblib first
        try:
            clusterer = joblib.load(filepath)
        except Exception as e:
            logger.warning(f"joblib load failed: {e}, trying pickle")
            with open(filepath, 'rb') as f:
                clusterer = pickle.load(f)

        return clusterer

    def load_features(self, run_id: int) -> tuple[npt.NDArray[np.float64], Dict[str, Any]]:
        """
        Load feature matrix for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            tuple: (features array, config dictionary)

        Raises:
            FileNotFoundError: If features file not found for run_id
        """
        # Find the features file for this run_id
        pattern = f"features_run{run_id:04d}_*.npz"
        matches = list(Config.LABELS_DIR.glob(pattern))

        if not matches:
            raise FileNotFoundError(f"No features file found for run_id {run_id}")

        if len(matches) > 1:
            logger.warning(f"Multiple features files found for run_id {run_id}, using first one")

        filepath = matches[0]
        logger.debug(f"Loading features from {filepath}")

        # Load from compressed numpy file
        data = np.load(filepath, allow_pickle=True)
        features = data['features']
        config = data['config'].item() if 'config' in data else {}

        return features, config

    def load_metrics_dataframe(self) -> pd.DataFrame:
        """
        Load all metrics from the CSV file.

        Returns:
            DataFrame with all run metrics

        Raises:
            FileNotFoundError: If metrics file doesn't exist

        Examples:
            >>> storage = ResultsStorage()
            >>> df = storage.load_metrics_dataframe()
            >>> print(f"Total runs: {len(df)}")
            >>> print(df.head())
        """
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")

        df = pd.read_csv(self.metrics_file)
        logger.info(f"Loaded metrics for {len(df)} runs")

        return df

    def get_best_runs(
        self,
        metric: str = 'silhouette_score',
        n: int = 10,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get the top N runs based on a specific metric.

        Args:
            metric: Metric to sort by (e.g., 'silhouette_score', 'davies_bouldin_score')
            n: Number of top runs to return
            ascending: If True, lower values are better (e.g., for davies_bouldin_score)

        Returns:
            DataFrame with top N runs

        Examples:
            >>> storage = ResultsStorage()
            >>> # Get top 5 runs by silhouette score (higher is better)
            >>> top_runs = storage.get_best_runs('silhouette_score', n=5, ascending=False)
            >>> # Get top 5 runs by davies_bouldin score (lower is better)
            >>> top_runs = storage.get_best_runs('davies_bouldin_score', n=5, ascending=True)
        """
        df = self.load_metrics_dataframe()

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in metrics. Available: {df.columns.tolist()}")

        # Sort by metric and get top N
        df_sorted = df.sort_values(by=metric, ascending=ascending)
        top_n = df_sorted.head(n)

        logger.info(f"Retrieved top {n} runs by {metric} (ascending={ascending})")

        return top_n

    def print_summary(self) -> None:
        """
        Print a summary of all stored results.

        Examples:
            >>> storage = ResultsStorage()
            >>> storage.print_summary()
        """
        try:
            df = self.load_metrics_dataframe()
        except FileNotFoundError:
            print("No results saved yet.")
            return

        print("\n" + "=" * 80)
        print("RESULTS STORAGE SUMMARY")
        print("=" * 80)

        print(f"\nTotal runs: {len(df)}")
        print(f"Metrics file: {self.metrics_file}")
        print(f"Labels directory: {Config.LABELS_DIR}")
        print(f"Models directory: {Config.MODELS_DIR}")

        if len(df) > 0:
            print("\nConfiguration summary:")
            print(f"  Window sizes: {sorted(df['window_size'].unique())}")
            print(f"  Min cluster sizes: {sorted(df['min_cluster_size'].unique())}")
            print(f"  Min samples: {sorted(df['min_samples'].unique())}")
            print(f"  Metrics: {sorted(df['metric'].unique())}")

            print("\nClustering results:")
            print(f"  Total clusters found: {df['n_clusters'].sum()}")
            print(f"  Average clusters per run: {df['n_clusters'].mean():.1f}")
            print(f"  Average noise ratio: {df['noise_ratio'].mean()*100:.1f}%")

            if 'silhouette_score' in df.columns and df['silhouette_score'].notna().any():
                print("\nQuality metrics (average):")
                print(f"  Silhouette score: {df['silhouette_score'].mean():.4f}")
                print(f"  Davies-Bouldin: {df['davies_bouldin_score'].mean():.4f}")
                print(f"  Calinski-Harabasz: {df['calinski_harabasz_score'].mean():.1f}")

                print("\nBest run (by silhouette score):")
                best_run = df.loc[df['silhouette_score'].idxmax()]
                print(f"  Run ID: {best_run['run_id']}")
                print(f"  Config: {best_run['config_id']}")
                print(f"  Silhouette: {best_run['silhouette_score']:.4f}")
                print(f"  Clusters: {best_run['n_clusters']}")

        print("=" * 80 + "\n")

    def delete_run(self, run_id: int) -> None:
        """
        Delete all files associated with a specific run.

        WARNING: This operation cannot be undone!

        Args:
            run_id: Run identifier to delete

        Examples:
            >>> storage = ResultsStorage()
            >>> storage.delete_run(run_id=5)
        """
        logger.warning(f"Deleting all files for run {run_id}")

        deleted_count = 0

        # Delete labels
        pattern = f"*_run{run_id:04d}_*.npz"
        for filepath in Config.LABELS_DIR.glob(pattern):
            filepath.unlink()
            logger.debug(f"Deleted {filepath}")
            deleted_count += 1

        # Delete clusterer
        pattern = f"*_run{run_id:04d}_*.pkl"
        for filepath in Config.MODELS_DIR.glob(pattern):
            filepath.unlink()
            logger.debug(f"Deleted {filepath}")
            deleted_count += 1

        # Remove from metrics CSV
        if self.metrics_file.exists():
            df = pd.read_csv(self.metrics_file)
            df_filtered = df[df['run_id'] != run_id]

            if len(df_filtered) < len(df):
                df_filtered.to_csv(self.metrics_file, index=False)
                logger.debug(f"Removed run {run_id} from metrics CSV")
                deleted_count += 1

        logger.info(f"Deleted {deleted_count} files for run {run_id}")


if __name__ == "__main__":
    # Test the ResultsStorage class
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing ResultsStorage module...\n")

    # Test 1: Initialize storage
    print("=" * 80)
    print("Test 1: Initialize ResultsStorage")
    print("=" * 80)
    storage = ResultsStorage()
    print(f"Output directory: {storage.output_dir}")
    print(f"Metrics file: {storage.metrics_file}")
    print(f"Next run ID: {storage.run_counter}\n")

    # Test 2: Create dummy data
    print("=" * 80)
    print("Test 2: Create dummy clustering results")
    print("=" * 80)
    np.random.seed(42)

    # Dummy configuration
    config = {
        'window_size': 10,
        'min_cluster_size': 5,
        'min_samples': 6,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom'
    }

    # Dummy labels (3 clusters + noise)
    labels = np.array([0] * 50 + [1] * 30 + [2] * 20 + [-1] * 10)
    np.random.shuffle(labels)

    # Dummy metrics
    metrics = {
        'n_clusters': 3,
        'n_noise_points': 10,
        'noise_ratio': 0.09,
        'cluster_labels': [0, 1, 2],
        'cluster_sizes': [50, 30, 20],
        'silhouette_score': 0.456,
        'davies_bouldin_score': 0.89,
        'calinski_harabasz_score': 123.45
    }

    # Dummy features
    features = np.random.rand(110, 40)

    print(f"Created dummy data:")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Config: {Config.get_config_id(config)}\n")

    # Test 3: Save results
    print("=" * 80)
    print("Test 3: Save results")
    print("=" * 80)
    run_id = storage.save_run_results(
        config=config,
        labels=labels,
        metrics=metrics,
        clusterer=None,  # Skip saving clusterer in test
        features=features,
        save_clusterer=False,
        save_features=True
    )
    print(f"Saved run with ID: {run_id}\n")

    # Test 4: Save another run with different config
    print("=" * 80)
    print("Test 4: Save second run with different configuration")
    print("=" * 80)
    config2 = config.copy()
    config2['window_size'] = 15
    config2['min_cluster_size'] = 10

    labels2 = np.array([0] * 60 + [1] * 40 + [-1] * 20)
    np.random.shuffle(labels2)

    metrics2 = {
        'n_clusters': 2,
        'n_noise_points': 20,
        'noise_ratio': 0.167,
        'cluster_labels': [0, 1],
        'cluster_sizes': [60, 40],
        'silhouette_score': 0.512,
        'davies_bouldin_score': 0.76,
        'calinski_harabasz_score': 145.67
    }

    run_id2 = storage.save_run_results(
        config=config2,
        labels=labels2,
        metrics=metrics2,
        save_clusterer=False,
        save_features=False
    )
    print(f"Saved run with ID: {run_id2}\n")

    # Test 5: Load results
    print("=" * 80)
    print("Test 5: Load saved results")
    print("=" * 80)
    loaded_labels, loaded_config = storage.load_labels(run_id)
    print(f"Loaded labels shape: {loaded_labels.shape}")
    print(f"Loaded config: {Config.get_config_id(loaded_config)}")
    print(f"Labels match: {np.array_equal(labels, loaded_labels)}")

    # Load features
    loaded_features, _ = storage.load_features(run_id)
    print(f"Loaded features shape: {loaded_features.shape}")
    print(f"Features match: {np.array_equal(features, loaded_features)}\n")

    # Test 6: Load metrics dataframe
    print("=" * 80)
    print("Test 6: Load metrics dataframe")
    print("=" * 80)
    df = storage.load_metrics_dataframe()
    print(f"Metrics DataFrame shape: {df.shape}")
    print(f"\nMetrics DataFrame:\n{df[['run_id', 'config_id', 'n_clusters', 'silhouette_score']]}\n")

    # Test 7: Get best runs
    print("=" * 80)
    print("Test 7: Get best runs")
    print("=" * 80)
    best_runs = storage.get_best_runs('silhouette_score', n=2, ascending=False)
    print(f"Top 2 runs by silhouette score:")
    print(best_runs[['run_id', 'config_id', 'n_clusters', 'silhouette_score']])
    print()

    # Test 8: Print summary
    print("=" * 80)
    print("Test 8: Print storage summary")
    print("=" * 80)
    storage.print_summary()

    # Test 9: Error handling
    print("=" * 80)
    print("Test 9: Test error handling")
    print("=" * 80)

    try:
        storage.load_labels(999)
    except FileNotFoundError as e:
        print(f"✓ Caught expected error for non-existent run: {e}")

    try:
        storage.load_clusterer(999)
    except FileNotFoundError as e:
        print(f"✓ Caught expected error for non-existent clusterer: {e}")

    print("\n✓ All ResultsStorage tests completed successfully!")
