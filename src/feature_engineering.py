"""Feature Engineering for OHLCV Windows

This module provides functionality to extract and transform features from
OHLCV windows for clustering analysis.
"""

import logging
from typing import Optional, Literal
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from OHLCV windows for clustering.

    The basic extraction flattens each window's OHLC values into a 1D feature vector.
    Additional feature engineering options may be added in the future (e.g., returns,
    normalized prices, technical indicators).

    Attributes:
        feature_type (str): Type of features to extract ('flatten', 'returns', etc.)
        flatten_order (str): Order for flattening ('sequential' or 'columnar')
    """

    def __init__(
        self,
        feature_type: Literal['flatten', 'returns', 'normalized'] = 'flatten',
        flatten_order: Literal['sequential', 'columnar'] = 'sequential'
    ):
        """
        Initialize the feature extractor.

        Args:
            feature_type: Type of feature extraction to perform
                - 'flatten': Direct flattening of OHLC values
                - 'returns': Calculate returns between bars (future enhancement)
                - 'normalized': Normalize to first bar (future enhancement)
            flatten_order: Order of flattening for 'flatten' mode
                - 'sequential': [bar0_O, bar0_H, bar0_L, bar0_C, bar1_O, bar1_H, ...]
                - 'columnar': [bar0_O, bar1_O, ..., barN_O, bar0_H, bar1_H, ...]
        """
        self.feature_type = feature_type
        self.flatten_order = flatten_order

        if feature_type not in ['flatten', 'returns', 'normalized']:
            raise ValueError(
                f"feature_type must be 'flatten', 'returns', or 'normalized', got '{feature_type}'"
            )

        if flatten_order not in ['sequential', 'columnar']:
            raise ValueError(
                f"flatten_order must be 'sequential' or 'columnar', got '{flatten_order}'"
            )

        logger.debug(
            f"FeatureExtractor initialized: type={feature_type}, "
            f"flatten_order={flatten_order}"
        )

    def extract_features(self, windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Extract features from OHLCV windows.

        Takes 3D array of windows and converts to 2D feature matrix suitable
        for clustering algorithms.

        Args:
            windows: Array of shape (n_windows, window_size, 4)
                    where 4 represents [Open, High, Low, Close]

        Returns:
            Array of shape (n_windows, n_features)
            - For 'flatten': n_features = window_size * 4
            - For 'returns': n_features = (window_size - 1) * 4
            - For 'normalized': n_features = window_size * 4

        Raises:
            ValueError: If windows array has invalid shape or contains invalid values

        Examples:
            >>> extractor = FeatureExtractor(feature_type='flatten')
            >>> windows = np.random.rand(100, 10, 4)  # 100 windows, size 10
            >>> features = extractor.extract_features(windows)
            >>> print(features.shape)
            (100, 40)  # 10 bars * 4 OHLC values
        """
        self._validate_windows(windows)

        if self.feature_type == 'flatten':
            features = self._extract_flatten(windows)
        elif self.feature_type == 'returns':
            features = self._extract_returns(windows)
        elif self.feature_type == 'normalized':
            features = self._extract_normalized(windows)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        logger.info(
            f"Extracted features: {windows.shape} -> {features.shape} "
            f"(type={self.feature_type})"
        )

        return features

    def _validate_windows(self, windows: npt.NDArray) -> None:
        """
        Validate windows array shape and values.

        Args:
            windows: Windows array to validate

        Raises:
            ValueError: If validation fails
        """
        if windows.ndim != 3:
            raise ValueError(
                f"Windows must be 3D array (n_windows, window_size, 4), "
                f"got {windows.ndim}D array with shape {windows.shape}"
            )

        if windows.shape[2] != 4:
            raise ValueError(
                f"Last dimension must be 4 (OHLC), got {windows.shape[2]}"
            )

        if windows.shape[0] == 0:
            raise ValueError("No windows provided (n_windows=0)")

        if windows.shape[1] == 0:
            raise ValueError("Window size cannot be 0")

        # Check for NaN or Inf values
        if np.any(np.isnan(windows)):
            n_nan = np.sum(np.isnan(windows))
            raise ValueError(f"Windows contain {n_nan} NaN values")

        if np.any(np.isinf(windows)):
            n_inf = np.sum(np.isinf(windows))
            raise ValueError(f"Windows contain {n_inf} infinite values")

        logger.debug(f"Windows validation passed: shape={windows.shape}")

    def _extract_flatten(self, windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Flatten OHLC values for each window.

        Args:
            windows: Array of shape (n_windows, window_size, 4)

        Returns:
            Array of shape (n_windows, window_size * 4)
        """
        n_windows, window_size, n_features = windows.shape

        if self.flatten_order == 'sequential':
            # Sequential: [bar0_O, bar0_H, bar0_L, bar0_C, bar1_O, bar1_H, ...]
            # This is the default numpy flatten behavior
            features = windows.reshape(n_windows, window_size * n_features)

        elif self.flatten_order == 'columnar':
            # Columnar: [bar0_O, bar1_O, ..., barN_O, bar0_H, bar1_H, ...]
            # Transpose to get (n_windows, 4, window_size) then flatten
            features = windows.transpose(0, 2, 1).reshape(n_windows, window_size * n_features)

        else:
            raise ValueError(f"Unknown flatten_order: {self.flatten_order}")

        logger.debug(
            f"Flattened {n_windows} windows: "
            f"{window_size} bars * {n_features} features = "
            f"{features.shape[1]} total features"
        )

        return features

    def _extract_returns(self, windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate returns between consecutive bars.

        Computes percentage returns for each OHLC value between consecutive bars.
        This reduces the feature count by 1 bar but may capture pattern dynamics better.

        Args:
            windows: Array of shape (n_windows, window_size, 4)

        Returns:
            Array of shape (n_windows, (window_size - 1) * 4)
        """
        n_windows, window_size, n_features = windows.shape

        # Calculate returns: (price[t] - price[t-1]) / price[t-1]
        # Result shape: (n_windows, window_size - 1, 4)
        returns = np.diff(windows, axis=1) / windows[:, :-1, :]

        # Flatten the returns
        if self.flatten_order == 'sequential':
            features = returns.reshape(n_windows, (window_size - 1) * n_features)
        else:
            features = returns.transpose(0, 2, 1).reshape(n_windows, (window_size - 1) * n_features)

        logger.debug(
            f"Extracted returns from {n_windows} windows: "
            f"{window_size - 1} bars * {n_features} features = "
            f"{features.shape[1]} total features"
        )

        return features

    def _extract_normalized(self, windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Normalize each window to its first bar's closing price.

        This creates price-independent patterns by normalizing all OHLC values
        to the starting price level.

        Args:
            windows: Array of shape (n_windows, window_size, 4)

        Returns:
            Array of shape (n_windows, window_size * 4)
        """
        n_windows, window_size, n_features = windows.shape

        # Normalize to first bar's close price (index 3 is Close)
        # Avoid division by zero
        first_close = windows[:, 0, 3:4]  # Shape: (n_windows, 1)
        first_close = np.where(first_close == 0, 1e-10, first_close)  # Prevent div by zero

        # Normalize all OHLC values by first close
        # Broadcasting: (n_windows, window_size, 4) / (n_windows, 1, 1)
        normalized = windows / first_close[:, np.newaxis, :]

        # Flatten normalized windows
        if self.flatten_order == 'sequential':
            features = normalized.reshape(n_windows, window_size * n_features)
        else:
            features = normalized.transpose(0, 2, 1).reshape(n_windows, window_size * n_features)

        logger.debug(
            f"Normalized {n_windows} windows to first bar close: "
            f"{window_size} bars * {n_features} features = "
            f"{features.shape[1]} total features"
        )

        return features

    def get_feature_names(self, window_size: int) -> list[str]:
        """
        Get feature names for the extracted features.

        Useful for debugging and interpretation.

        Args:
            window_size: Size of the windows used

        Returns:
            List of feature names

        Examples:
            >>> extractor = FeatureExtractor(flatten_order='sequential')
            >>> names = extractor.get_feature_names(window_size=3)
            >>> print(names[:8])
            ['bar0_Open', 'bar0_High', 'bar0_Low', 'bar0_Close',
             'bar1_Open', 'bar1_High', 'bar1_Low', 'bar1_Close']
        """
        ohlc_names = ['Open', 'High', 'Low', 'Close']

        if self.feature_type == 'flatten':
            n_bars = window_size
            prefix = 'bar'
        elif self.feature_type == 'returns':
            n_bars = window_size - 1
            prefix = 'return'
        elif self.feature_type == 'normalized':
            n_bars = window_size
            prefix = 'norm_bar'
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        if self.flatten_order == 'sequential':
            # [bar0_Open, bar0_High, bar0_Low, bar0_Close, bar1_Open, ...]
            names = [
                f"{prefix}{i}_{col}"
                for i in range(n_bars)
                for col in ohlc_names
            ]
        else:
            # [bar0_Open, bar1_Open, ..., bar0_High, bar1_High, ...]
            names = [
                f"{prefix}{i}_{col}"
                for col in ohlc_names
                for i in range(n_bars)
            ]

        return names


if __name__ == "__main__":
    # Test the feature extractor
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing Feature Extractor...\n")

    # Create sample windows (5 windows, 10 bars each, 4 OHLC values)
    print("Creating sample windows...")
    np.random.seed(42)
    n_windows, window_size, n_ohlc = 5, 10, 4
    windows = np.random.rand(n_windows, window_size, n_ohlc) * 100 + 100
    print(f"Windows shape: {windows.shape}\n")

    # Test flatten extraction with sequential order
    print("=" * 70)
    print("Test 1: Flatten with sequential order")
    print("=" * 70)
    extractor_seq = FeatureExtractor(feature_type='flatten', flatten_order='sequential')
    features_seq = extractor_seq.extract_features(windows)
    print(f"Features shape: {features_seq.shape}")
    print(f"Expected shape: ({n_windows}, {window_size * n_ohlc})")
    print(f"First 8 features of window 0: {features_seq[0, :8]}")
    print(f"Expected (first 2 bars, sequential): {windows[0, :2, :].flatten()}")
    print(f"Match: {np.allclose(features_seq[0, :8], windows[0, :2, :].flatten())}\n")

    # Test flatten extraction with columnar order
    print("=" * 70)
    print("Test 2: Flatten with columnar order")
    print("=" * 70)
    extractor_col = FeatureExtractor(feature_type='flatten', flatten_order='columnar')
    features_col = extractor_col.extract_features(windows)
    print(f"Features shape: {features_col.shape}")
    print(f"First 10 features (all Opens): {features_col[0, :10]}")
    print(f"Expected (all Opens): {windows[0, :, 0]}")
    print(f"Match: {np.allclose(features_col[0, :10], windows[0, :, 0])}\n")

    # Test returns extraction
    print("=" * 70)
    print("Test 3: Returns extraction")
    print("=" * 70)
    extractor_ret = FeatureExtractor(feature_type='returns')
    features_ret = extractor_ret.extract_features(windows)
    print(f"Features shape: {features_ret.shape}")
    print(f"Expected shape: ({n_windows}, {(window_size - 1) * n_ohlc})")
    print(f"First 4 return features (bar0->bar1 returns):")
    expected_returns = (windows[0, 1, :] - windows[0, 0, :]) / windows[0, 0, :]
    print(f"  Extracted: {features_ret[0, :4]}")
    print(f"  Expected: {expected_returns}")
    print(f"  Match: {np.allclose(features_ret[0, :4], expected_returns)}\n")

    # Test normalized extraction
    print("=" * 70)
    print("Test 4: Normalized extraction")
    print("=" * 70)
    extractor_norm = FeatureExtractor(feature_type='normalized')
    features_norm = extractor_norm.extract_features(windows)
    print(f"Features shape: {features_norm.shape}")
    print(f"First bar's close price: {windows[0, 0, 3]}")
    print(f"Normalized first 4 values: {features_norm[0, :4]}")
    expected_norm = windows[0, 0, :] / windows[0, 0, 3]
    print(f"Expected normalized: {expected_norm}")
    print(f"Match: {np.allclose(features_norm[0, :4], expected_norm)}\n")

    # Test feature names
    print("=" * 70)
    print("Test 5: Feature names")
    print("=" * 70)
    names_seq = extractor_seq.get_feature_names(window_size)
    names_col = extractor_col.get_feature_names(window_size)
    print(f"Sequential order (first 8): {names_seq[:8]}")
    print(f"Columnar order (first 12): {names_col[:12]}")

    # Test error handling
    print("\n" + "=" * 70)
    print("Test 6: Error handling")
    print("=" * 70)

    try:
        bad_windows = np.random.rand(5, 10, 3)  # Wrong last dimension
        extractor_seq.extract_features(bad_windows)
    except ValueError as e:
        print(f"✓ Caught expected error for wrong dimension: {e}")

    try:
        nan_windows = windows.copy()
        nan_windows[0, 0, 0] = np.nan
        extractor_seq.extract_features(nan_windows)
    except ValueError as e:
        print(f"✓ Caught expected error for NaN values: {e}")

    print("\n✓ All feature extraction tests completed successfully!")
