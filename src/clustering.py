"""HDBSCAN Clustering Wrapper with GPU/CPU Support

This module provides a unified interface for HDBSCAN clustering that works
with both GPU (cuML) and CPU (hdbscan) backends seamlessly.
"""

import logging
from typing import Tuple, Any, Optional, Dict
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class HDBSCANClusterer:
    """
    Unified interface for GPU/CPU HDBSCAN clustering with automatic fallback.

    This class wraps both cuML's GPU-accelerated HDBSCAN and the standard
    CPU-based hdbscan library, providing a consistent API regardless of backend.

    Attributes:
        backend_type (str): Type of backend ('gpu' or 'cpu')
        backend_module: The HDBSCAN module being used (cuml.cluster or hdbscan)
        clusterer: The fitted HDBSCAN clusterer object (after fit_predict)
        labels_ (np.ndarray): Cluster labels from last fit (after fit_predict)
    """

    def __init__(self, backend_type: str, backend_module: Any):
        """
        Initialize the HDBSCAN clusterer with specified backend.

        Args:
            backend_type: Type of backend ('gpu' or 'cpu')
            backend_module: The module to use (cuml.cluster or hdbscan)

        Examples:
            >>> from src.gpu_utils import detect_compute_backend
            >>> backend_type, backend_module = detect_compute_backend()
            >>> clusterer = HDBSCANClusterer(backend_type, backend_module)
        """
        if backend_type not in ['gpu', 'cpu']:
            raise ValueError(f"backend_type must be 'gpu' or 'cpu', got '{backend_type}'")

        self.backend_type = backend_type
        self.backend_module = backend_module
        self.clusterer: Optional[Any] = None
        self.labels_: Optional[npt.NDArray[np.int32]] = None

        logger.info(f"HDBSCANClusterer initialized with {backend_type.upper()} backend")

    def fit_predict(
        self,
        features: npt.NDArray[np.float64],
        min_cluster_size: int,
        min_samples: int,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom',
        **kwargs
    ) -> Tuple[npt.NDArray[np.int32], Any]:
        """
        Run HDBSCAN clustering on features.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in a neighborhood for a point to be core
            metric: Distance metric to use (default: 'euclidean')
            cluster_selection_method: Method for selecting clusters from tree
                - 'eom': Excess of Mass (default)
                - 'leaf': Select leaf nodes
            **kwargs: Additional parameters to pass to HDBSCAN

        Returns:
            tuple: (labels, clusterer_object)
                - labels: Array of cluster labels (-1 for noise)
                - clusterer_object: Fitted HDBSCAN object with tree information

        Raises:
            ValueError: If parameters are invalid or features have issues
            RuntimeError: If clustering fails

        Examples:
            >>> features = np.random.rand(1000, 40)
            >>> labels, clusterer = hdbscan_obj.fit_predict(
            ...     features,
            ...     min_cluster_size=5,
            ...     min_samples=6
            ... )
            >>> print(f"Found {labels.max() + 1} clusters")
        """
        # Validate inputs
        self._validate_inputs(features, min_cluster_size, min_samples)

        logger.info(
            f"Running HDBSCAN clustering: "
            f"backend={self.backend_type}, "
            f"n_samples={features.shape[0]}, "
            f"n_features={features.shape[1]}, "
            f"min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, "
            f"metric={metric}"
        )

        try:
            if self.backend_type == 'gpu':
                labels, clusterer = self._fit_predict_gpu(
                    features, min_cluster_size, min_samples, metric,
                    cluster_selection_method, **kwargs
                )
            else:
                labels, clusterer = self._fit_predict_cpu(
                    features, min_cluster_size, min_samples, metric,
                    cluster_selection_method, **kwargs
                )

            # Store results
            self.clusterer = clusterer
            self.labels_ = labels

            # Log clustering results
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            logger.info(
                f"Clustering complete: {n_clusters} clusters found, "
                f"{n_noise} noise points ({n_noise/len(labels)*100:.1f}%)"
            )

            return labels, clusterer

        except Exception as e:
            logger.error(f"Clustering failed with {self.backend_type} backend: {e}")
            raise RuntimeError(f"HDBSCAN clustering failed: {e}") from e

    def _validate_inputs(
        self,
        features: npt.NDArray,
        min_cluster_size: int,
        min_samples: int
    ) -> None:
        """
        Validate input parameters for clustering.

        Args:
            features: Feature matrix
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples parameter

        Raises:
            ValueError: If validation fails
        """
        # Check features array
        if features.ndim != 2:
            raise ValueError(
                f"Features must be 2D array (n_samples, n_features), "
                f"got {features.ndim}D array with shape {features.shape}"
            )

        if features.shape[0] == 0:
            raise ValueError("No samples provided (n_samples=0)")

        if features.shape[1] == 0:
            raise ValueError("No features provided (n_features=0)")

        # Check for invalid values
        if np.any(np.isnan(features)):
            n_nan = np.sum(np.isnan(features))
            raise ValueError(f"Features contain {n_nan} NaN values")

        if np.any(np.isinf(features)):
            n_inf = np.sum(np.isinf(features))
            raise ValueError(f"Features contain {n_inf} infinite values")

        # Check parameters
        if min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")

        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")

        if min_samples > min_cluster_size:
            logger.warning(
                f"min_samples ({min_samples}) > min_cluster_size ({min_cluster_size}). "
                f"This may result in no clusters being found."
            )

        logger.debug(
            f"Input validation passed: "
            f"shape={features.shape}, "
            f"min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}"
        )

    def _fit_predict_gpu(
        self,
        features: npt.NDArray[np.float64],
        min_cluster_size: int,
        min_samples: int,
        metric: str,
        cluster_selection_method: str,
        **kwargs
    ) -> Tuple[npt.NDArray[np.int32], Any]:
        """
        Run HDBSCAN on GPU using cuML.

        Args:
            features: Feature matrix (numpy array)
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples parameter
            metric: Distance metric
            cluster_selection_method: Cluster selection method
            **kwargs: Additional parameters

        Returns:
            tuple: (labels, clusterer) - labels as numpy array
        """
        import cupy as cp

        logger.debug("Converting features to CuPy array for GPU processing")
        features_gpu = cp.asarray(features, dtype=cp.float32)

        # Initialize cuML HDBSCAN
        clusterer = self.backend_module.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            **kwargs
        )

        # Fit and predict
        logger.debug("Running GPU HDBSCAN fit_predict...")
        clusterer.fit(features_gpu)

        # Get labels and convert back to numpy
        labels_gpu = clusterer.labels_
        labels = cp.asnumpy(labels_gpu).astype(np.int32)

        logger.debug("Converting GPU results back to CPU")

        return labels, clusterer

    def _fit_predict_cpu(
        self,
        features: npt.NDArray[np.float64],
        min_cluster_size: int,
        min_samples: int,
        metric: str,
        cluster_selection_method: str,
        **kwargs
    ) -> Tuple[npt.NDArray[np.int32], Any]:
        """
        Run HDBSCAN on CPU using standard hdbscan library.

        Args:
            features: Feature matrix (numpy array)
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples parameter
            metric: Distance metric
            cluster_selection_method: Cluster selection method
            **kwargs: Additional parameters

        Returns:
            tuple: (labels, clusterer)
        """
        # Enable prediction data by default if not specified
        if 'prediction_data' not in kwargs:
            kwargs['prediction_data'] = True
            logger.debug("Enabled prediction_data=True for future predictions")

        # Initialize CPU HDBSCAN
        clusterer = self.backend_module.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            **kwargs
        )

        # Fit and predict
        logger.debug("Running CPU HDBSCAN fit_predict...")
        clusterer.fit(features)
        labels = clusterer.labels_.astype(np.int32)

        return labels, clusterer

    def get_cluster_tree(self) -> Optional[Any]:
        """
        Get the condensed cluster tree from the fitted clusterer.

        The tree structure is useful for visualization and understanding
        the hierarchical clustering process.

        Returns:
            Condensed tree object if available, None otherwise

        Raises:
            RuntimeError: If clustering has not been performed yet

        Examples:
            >>> labels, _ = clusterer.fit_predict(features, 5, 6)
            >>> tree = clusterer.get_cluster_tree()
            >>> if tree is not None:
            ...     # Plot or analyze tree
            ...     pass
        """
        if self.clusterer is None:
            raise RuntimeError("Must call fit_predict() before accessing cluster tree")

        if hasattr(self.clusterer, 'condensed_tree_'):
            return self.clusterer.condensed_tree_
        else:
            logger.warning(
                f"Condensed tree not available for {self.backend_type} backend"
            )
            return None

    def get_cluster_persistence(self) -> Optional[npt.NDArray]:
        """
        Get cluster persistence values (measure of cluster stability).

        Returns:
            Array of persistence values if available, None otherwise
        """
        if self.clusterer is None:
            raise RuntimeError("Must call fit_predict() before accessing persistence")

        if hasattr(self.clusterer, 'cluster_persistence_'):
            return self.clusterer.cluster_persistence_
        else:
            logger.warning(
                f"Cluster persistence not available for {self.backend_type} backend"
            )
            return None

    def get_probabilities(self) -> Optional[npt.NDArray]:
        """
        Get cluster membership probabilities for each point.

        Returns:
            Array of probabilities if available, None otherwise
        """
        if self.clusterer is None:
            raise RuntimeError("Must call fit_predict() before accessing probabilities")

        if hasattr(self.clusterer, 'probabilities_'):
            probs = self.clusterer.probabilities_
            if self.backend_type == 'gpu':
                import cupy as cp
                return cp.asnumpy(probs)
            return probs
        else:
            logger.warning(
                f"Probabilities not available for {self.backend_type} backend"
            )
            return None

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the clusterer configuration and results.

        Returns:
            Dictionary with clusterer information

        Examples:
            >>> info = clusterer.get_info()
            >>> print(f"Backend: {info['backend_type']}")
            >>> print(f"Clusters found: {info['n_clusters']}")
        """
        info = {
            'backend_type': self.backend_type,
            'backend_module': self.backend_module.__name__,
            'fitted': self.clusterer is not None,
        }

        if self.labels_ is not None:
            unique_labels = set(self.labels_)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(self.labels_ == -1)

            info.update({
                'n_samples': len(self.labels_),
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(self.labels_),
                'cluster_labels': sorted([l for l in unique_labels if l != -1])
            })

        return info


if __name__ == "__main__":
    # Test the HDBSCAN wrapper
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing HDBSCAN Clusterer...\n")

    # Import GPU detection
    from gpu_utils import detect_compute_backend

    # Detect backend
    print("=" * 70)
    print("Step 1: Detect compute backend")
    print("=" * 70)
    backend_type, backend_module = detect_compute_backend()
    print(f"Backend: {backend_type}")
    print(f"Module: {backend_module.__name__}\n")

    # Initialize clusterer
    print("=" * 70)
    print("Step 2: Initialize clusterer")
    print("=" * 70)
    clusterer = HDBSCANClusterer(backend_type, backend_module)
    print(f"Clusterer initialized: {clusterer.get_info()}\n")

    # Create test features
    print("=" * 70)
    print("Step 3: Create synthetic test data")
    print("=" * 70)
    np.random.seed(42)

    # Generate 3 clusters + noise
    n_samples_per_cluster = 100
    n_features = 40  # Simulating 10-bar window with 4 OHLC values

    # Cluster 1: centered at (0, 0, ...)
    cluster1 = np.random.randn(n_samples_per_cluster, n_features) * 0.5

    # Cluster 2: centered at (5, 5, ...)
    cluster2 = np.random.randn(n_samples_per_cluster, n_features) * 0.5 + 5

    # Cluster 3: centered at (-5, 5, ...)
    cluster3 = np.random.randn(n_samples_per_cluster, n_features) * 0.5
    cluster3[:, :10] -= 5
    cluster3[:, 10:] += 5

    # Noise: uniformly distributed
    noise = np.random.uniform(-10, 10, (50, n_features))

    # Combine all data
    features = np.vstack([cluster1, cluster2, cluster3, noise])
    print(f"Created test dataset: {features.shape}")
    print(f"  - 3 clusters of {n_samples_per_cluster} points each")
    print(f"  - 50 noise points")
    print(f"  - {n_features} features\n")

    # Run clustering
    print("=" * 70)
    print("Step 4: Run HDBSCAN clustering")
    print("=" * 70)
    min_cluster_size = 30
    min_samples = 10

    labels, fitted_clusterer = clusterer.fit_predict(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )

    print(f"\nClustering results:")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Unique labels: {sorted(set(labels))}")

    # Get cluster info
    print("\n" + "=" * 70)
    print("Step 5: Analyze clustering results")
    print("=" * 70)
    info = clusterer.get_info()
    print(f"Clusters found: {info['n_clusters']}")
    print(f"Noise points: {info['n_noise']} ({info['noise_ratio']*100:.1f}%)")
    print(f"Cluster labels: {info['cluster_labels']}")

    # Print cluster sizes
    for cluster_id in info['cluster_labels']:
        size = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id}: {size} points")

    # Test additional methods
    print("\n" + "=" * 70)
    print("Step 6: Test additional methods")
    print("=" * 70)

    tree = clusterer.get_cluster_tree()
    print(f"Condensed tree available: {tree is not None}")

    persistence = clusterer.get_cluster_persistence()
    print(f"Cluster persistence available: {persistence is not None}")
    if persistence is not None:
        print(f"  Persistence values: {persistence}")

    probabilities = clusterer.get_probabilities()
    print(f"Cluster probabilities available: {probabilities is not None}")
    if probabilities is not None:
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Mean probability: {probabilities.mean():.3f}")
        print(f"  Min/Max: {probabilities.min():.3f} / {probabilities.max():.3f}")

    # Test error handling
    print("\n" + "=" * 70)
    print("Step 7: Test error handling")
    print("=" * 70)

    try:
        bad_features = np.random.rand(10, 5, 3)  # 3D array
        clusterer.fit_predict(bad_features, 5, 3)
    except ValueError as e:
        print(f"✓ Caught expected error for wrong dimensions: {str(e)[:60]}...")

    try:
        nan_features = features.copy()
        nan_features[0, 0] = np.nan
        clusterer.fit_predict(nan_features, 5, 3)
    except ValueError as e:
        print(f"✓ Caught expected error for NaN values: {str(e)[:60]}...")

    try:
        clusterer.fit_predict(features, min_cluster_size=1, min_samples=3)
    except ValueError as e:
        print(f"✓ Caught expected error for invalid min_cluster_size: {str(e)[:60]}...")

    print("\n✓ All HDBSCAN clusterer tests completed successfully!")
