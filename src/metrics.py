"""Clustering Quality Metrics

This module provides utilities for computing and analyzing clustering quality metrics
including silhouette score, Davies-Bouldin index, and Calinski-Harabasz score.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

logger = logging.getLogger(__name__)


class ClusterMetrics:
    """
    Computes and stores clustering quality metrics.

    This class provides methods to calculate various metrics that evaluate
    the quality of clustering results, including:
    - Basic cluster statistics (counts, sizes, noise ratio)
    - Silhouette score (cluster separation)
    - Davies-Bouldin index (cluster compactness)
    - Calinski-Harabasz score (cluster variance ratio)

    Examples:
        >>> labels = np.array([0, 0, 1, 1, -1])
        >>> features = np.random.rand(5, 10)
        >>> metrics = ClusterMetrics.compute_metrics(labels, features)
        >>> print(f"Found {metrics['n_clusters']} clusters")
        >>> print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
    """

    @staticmethod
    def compute_metrics(
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float64],
        max_samples_for_silhouette: int = 10000
    ) -> Dict[str, Any]:
        """
        Compute comprehensive clustering quality metrics.

        Args:
            labels: Cluster labels array where -1 indicates noise points
            features: Feature matrix of shape (n_samples, n_features)
            max_samples_for_silhouette: Maximum samples to use for silhouette score
                computation (sampling used if exceeded for performance)

        Returns:
            Dictionary containing:
                - n_clusters (int): Number of clusters (excluding noise)
                - n_noise_points (int): Number of noise points (label == -1)
                - noise_ratio (float): Fraction of points classified as noise
                - cluster_sizes (List[int]): Size of each cluster
                - cluster_labels (List[int]): Unique cluster labels (excluding -1)
                - silhouette_score (float | None): Silhouette coefficient
                - davies_bouldin_score (float | None): Davies-Bouldin index
                - calinski_harabasz_score (float | None): Calinski-Harabasz score

        Raises:
            ValueError: If inputs are invalid or shapes don't match

        Examples:
            >>> labels = np.array([0, 0, 0, 1, 1, 1, -1, -1])
            >>> features = np.random.rand(8, 20)
            >>> metrics = ClusterMetrics.compute_metrics(labels, features)
            >>> print(f"Clusters: {metrics['n_clusters']}")
            >>> print(f"Noise ratio: {metrics['noise_ratio']:.2%}")
        """
        # Validate inputs
        ClusterMetrics._validate_inputs(labels, features)

        logger.info(
            f"Computing metrics for {len(labels)} samples with "
            f"{features.shape[1]} features"
        )

        # Initialize metrics dictionary
        metrics: Dict[str, Any] = {}

        # Basic statistics
        unique_labels = set(labels)
        cluster_labels = sorted([label for label in unique_labels if label != -1])
        n_clusters = len(cluster_labels)
        n_noise = int(np.sum(labels == -1))
        n_total = len(labels)

        metrics['n_clusters'] = n_clusters
        metrics['n_noise_points'] = n_noise
        metrics['noise_ratio'] = n_noise / n_total if n_total > 0 else 0.0
        metrics['cluster_labels'] = cluster_labels

        # Cluster sizes
        cluster_sizes = [int(np.sum(labels == label)) for label in cluster_labels]
        metrics['cluster_sizes'] = cluster_sizes

        logger.debug(
            f"Basic stats: {n_clusters} clusters, {n_noise} noise points "
            f"({metrics['noise_ratio']*100:.1f}%)"
        )

        # Advanced metrics require at least 2 clusters
        if n_clusters < 2:
            logger.warning(
                f"Cannot compute advanced metrics with {n_clusters} cluster(s). "
                f"At least 2 clusters required."
            )
            metrics['silhouette_score'] = None
            metrics['davies_bouldin_score'] = None
            metrics['calinski_harabasz_score'] = None
            return metrics

        # Filter out noise points for sklearn metrics
        non_noise_mask = labels != -1
        labels_no_noise = labels[non_noise_mask]
        features_no_noise = features[non_noise_mask]

        # Check if we have enough points after filtering noise
        if len(labels_no_noise) < 2:
            logger.warning("Not enough non-noise points for metrics computation")
            metrics['silhouette_score'] = None
            metrics['davies_bouldin_score'] = None
            metrics['calinski_harabasz_score'] = None
            return metrics

        # Compute silhouette score
        metrics['silhouette_score'] = ClusterMetrics._compute_silhouette_score(
            labels_no_noise,
            features_no_noise,
            max_samples_for_silhouette
        )

        # Compute Davies-Bouldin score
        metrics['davies_bouldin_score'] = ClusterMetrics._compute_davies_bouldin_score(
            labels_no_noise,
            features_no_noise
        )

        # Compute Calinski-Harabasz score
        metrics['calinski_harabasz_score'] = ClusterMetrics._compute_calinski_harabasz_score(
            labels_no_noise,
            features_no_noise
        )

        logger.info(
            f"Metrics computed: silhouette={metrics['silhouette_score']:.3f}, "
            f"davies_bouldin={metrics['davies_bouldin_score']:.3f}, "
            f"calinski_harabasz={metrics['calinski_harabasz_score']:.1f}"
        )

        return metrics

    @staticmethod
    def _validate_inputs(
        labels: npt.NDArray,
        features: npt.NDArray
    ) -> None:
        """
        Validate input arrays for metrics computation.

        Args:
            labels: Cluster labels array
            features: Feature matrix

        Raises:
            ValueError: If validation fails
        """
        # Check labels
        if labels.ndim != 1:
            raise ValueError(
                f"Labels must be 1D array, got {labels.ndim}D array with shape {labels.shape}"
            )

        # Check features
        if features.ndim != 2:
            raise ValueError(
                f"Features must be 2D array, got {features.ndim}D array with shape {features.shape}"
            )

        # Check matching lengths
        if len(labels) != features.shape[0]:
            raise ValueError(
                f"Labels length ({len(labels)}) must match features rows ({features.shape[0]})"
            )

        # Check for invalid values
        if np.any(np.isnan(features)):
            n_nan = np.sum(np.isnan(features))
            raise ValueError(f"Features contain {n_nan} NaN values")

        if np.any(np.isinf(features)):
            n_inf = np.sum(np.isinf(features))
            raise ValueError(f"Features contain {n_inf} infinite values")

        logger.debug(f"Input validation passed: {len(labels)} labels, {features.shape} features")

    @staticmethod
    def _compute_silhouette_score(
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float64],
        max_samples: int
    ) -> Optional[float]:
        """
        Compute silhouette score with optional sampling for large datasets.

        The silhouette score ranges from -1 to 1:
        - 1: Clusters are well separated
        - 0: Clusters are overlapping
        - -1: Points may be assigned to wrong clusters

        Args:
            labels: Cluster labels (no noise points)
            features: Feature matrix (no noise points)
            max_samples: Maximum samples to use (sample if exceeded)

        Returns:
            Silhouette score or None if computation fails
        """
        try:
            n_samples = len(labels)

            # Sample if dataset is too large
            if n_samples > max_samples:
                logger.debug(
                    f"Sampling {max_samples} points for silhouette score "
                    f"(total: {n_samples})"
                )
                sample_indices = np.random.choice(
                    n_samples,
                    size=max_samples,
                    replace=False
                )
                labels_sample = labels[sample_indices]
                features_sample = features[sample_indices]
            else:
                labels_sample = labels
                features_sample = features

            # Compute silhouette score
            score = silhouette_score(
                features_sample,
                labels_sample,
                metric='euclidean'
            )

            logger.debug(f"Silhouette score computed: {score:.4f}")
            return float(score)

        except Exception as e:
            logger.error(f"Failed to compute silhouette score: {e}")
            return None

    @staticmethod
    def _compute_davies_bouldin_score(
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float64]
    ) -> Optional[float]:
        """
        Compute Davies-Bouldin index.

        The Davies-Bouldin index measures cluster separation and compactness:
        - Lower values indicate better clustering (minimum is 0)
        - Represents average similarity between each cluster and its most similar cluster

        Args:
            labels: Cluster labels (no noise points)
            features: Feature matrix (no noise points)

        Returns:
            Davies-Bouldin score or None if computation fails
        """
        try:
            score = davies_bouldin_score(features, labels)
            logger.debug(f"Davies-Bouldin score computed: {score:.4f}")
            return float(score)

        except Exception as e:
            logger.error(f"Failed to compute Davies-Bouldin score: {e}")
            return None

    @staticmethod
    def _compute_calinski_harabasz_score(
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float64]
    ) -> Optional[float]:
        """
        Compute Calinski-Harabasz index (Variance Ratio Criterion).

        The Calinski-Harabasz index is the ratio of between-cluster dispersion
        to within-cluster dispersion:
        - Higher values indicate better defined clusters
        - No bounded range (higher is better)

        Args:
            labels: Cluster labels (no noise points)
            features: Feature matrix (no noise points)

        Returns:
            Calinski-Harabasz score or None if computation fails
        """
        try:
            score = calinski_harabasz_score(features, labels)
            logger.debug(f"Calinski-Harabasz score computed: {score:.2f}")
            return float(score)

        except Exception as e:
            logger.error(f"Failed to compute Calinski-Harabasz score: {e}")
            return None

    @staticmethod
    def print_metrics_summary(metrics: Dict[str, Any]) -> None:
        """
        Print a formatted summary of clustering metrics.

        Args:
            metrics: Dictionary of metrics from compute_metrics()

        Examples:
            >>> metrics = ClusterMetrics.compute_metrics(labels, features)
            >>> ClusterMetrics.print_metrics_summary(metrics)
        """
        print("\n" + "=" * 70)
        print("CLUSTERING METRICS SUMMARY")
        print("=" * 70)

        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Total clusters:     {metrics['n_clusters']}")
        print(f"  Noise points:       {metrics['n_noise_points']}")
        print(f"  Noise ratio:        {metrics['noise_ratio']*100:.2f}%")

        # Cluster sizes
        if metrics['cluster_sizes']:
            print(f"\nCluster Sizes:")
            for i, (label, size) in enumerate(zip(metrics['cluster_labels'], metrics['cluster_sizes'])):
                print(f"  Cluster {label:2d}:        {size:5d} points")

            if metrics['cluster_sizes']:
                print(f"\n  Min cluster size:   {min(metrics['cluster_sizes'])}")
                print(f"  Max cluster size:   {max(metrics['cluster_sizes'])}")
                print(f"  Mean cluster size:  {np.mean(metrics['cluster_sizes']):.1f}")

        # Quality metrics
        print(f"\nQuality Metrics:")

        if metrics['silhouette_score'] is not None:
            print(f"  Silhouette score:   {metrics['silhouette_score']:7.4f}  (higher is better, range: [-1, 1])")
        else:
            print(f"  Silhouette score:   N/A")

        if metrics['davies_bouldin_score'] is not None:
            print(f"  Davies-Bouldin:     {metrics['davies_bouldin_score']:7.4f}  (lower is better, min: 0)")
        else:
            print(f"  Davies-Bouldin:     N/A")

        if metrics['calinski_harabasz_score'] is not None:
            print(f"  Calinski-Harabasz:  {metrics['calinski_harabasz_score']:7.1f}  (higher is better)")
        else:
            print(f"  Calinski-Harabasz:  N/A")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test the ClusterMetrics class
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing ClusterMetrics module...\n")

    # Test Case 1: Normal clustering with multiple clusters
    print("=" * 70)
    print("Test Case 1: Normal clustering (3 clusters + noise)")
    print("=" * 70)

    np.random.seed(42)
    n_samples_per_cluster = 100
    n_features = 40

    # Generate 3 distinct clusters
    cluster1 = np.random.randn(n_samples_per_cluster, n_features) * 0.5
    cluster2 = np.random.randn(n_samples_per_cluster, n_features) * 0.5 + 5
    cluster3 = np.random.randn(n_samples_per_cluster, n_features) * 0.5 - 5

    # Combine features
    features = np.vstack([cluster1, cluster2, cluster3])

    # Create labels: 0, 1, 2 for clusters, -1 for noise
    labels = np.array(
        [0] * n_samples_per_cluster +
        [1] * n_samples_per_cluster +
        [2] * n_samples_per_cluster
    )

    # Add some noise points
    noise_indices = np.random.choice(len(labels), size=30, replace=False)
    labels[noise_indices] = -1

    print(f"Generated data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Labels: {len(labels)} total, {np.sum(labels == -1)} noise\n")

    # Compute metrics
    metrics = ClusterMetrics.compute_metrics(labels, features)
    ClusterMetrics.print_metrics_summary(metrics)

    # Test Case 2: Single cluster (edge case)
    print("\n" + "=" * 70)
    print("Test Case 2: Single cluster (edge case)")
    print("=" * 70)

    features_single = np.random.randn(100, 40)
    labels_single = np.array([0] * 90 + [-1] * 10)

    metrics_single = ClusterMetrics.compute_metrics(labels_single, features_single)
    ClusterMetrics.print_metrics_summary(metrics_single)

    # Test Case 3: All noise (edge case)
    print("\n" + "=" * 70)
    print("Test Case 3: All noise (edge case)")
    print("=" * 70)

    features_noise = np.random.randn(50, 40)
    labels_noise = np.array([-1] * 50)

    metrics_noise = ClusterMetrics.compute_metrics(labels_noise, features_noise)
    ClusterMetrics.print_metrics_summary(metrics_noise)

    # Test Case 4: Large dataset with sampling
    print("\n" + "=" * 70)
    print("Test Case 4: Large dataset (silhouette sampling)")
    print("=" * 70)

    # Generate large dataset
    n_large = 15000
    features_large = np.random.randn(n_large, 40)

    # Create 5 clusters
    cluster_size = n_large // 6
    labels_large = np.array(
        [0] * cluster_size +
        [1] * cluster_size +
        [2] * cluster_size +
        [3] * cluster_size +
        [4] * cluster_size +
        [-1] * (n_large - 5 * cluster_size)
    )

    print(f"Generated large dataset: {n_large} samples")
    print(f"Silhouette computation will use sampling (max 10000 samples)\n")

    metrics_large = ClusterMetrics.compute_metrics(
        labels_large,
        features_large,
        max_samples_for_silhouette=10000
    )
    ClusterMetrics.print_metrics_summary(metrics_large)

    # Test Case 5: Error handling
    print("\n" + "=" * 70)
    print("Test Case 5: Error handling")
    print("=" * 70)

    try:
        # Mismatched shapes
        bad_labels = np.array([0, 1, 2])
        bad_features = np.random.rand(5, 10)
        ClusterMetrics.compute_metrics(bad_labels, bad_features)
    except ValueError as e:
        print(f"✓ Caught expected error for mismatched shapes: {str(e)[:60]}...")

    try:
        # Wrong dimensions
        bad_labels_2d = np.array([[0, 1], [2, 3]])
        features_ok = np.random.rand(4, 10)
        ClusterMetrics.compute_metrics(bad_labels_2d, features_ok)
    except ValueError as e:
        print(f"✓ Caught expected error for wrong label dimensions: {str(e)[:60]}...")

    try:
        # NaN values
        nan_features = np.random.rand(10, 5)
        nan_features[0, 0] = np.nan
        labels_ok = np.array([0, 0, 1, 1, 2, 2, -1, -1, -1, -1])
        ClusterMetrics.compute_metrics(labels_ok, nan_features)
    except ValueError as e:
        print(f"✓ Caught expected error for NaN values: {str(e)[:60]}...")

    print("\n✓ All ClusterMetrics tests completed successfully!")
