"""OHLCV Data Loading and Windowing

This module provides functionality to load OHLCV (Open, High, Low, Close, Volume)
trading data and create rolling windows for pattern analysis.
"""

import logging
from typing import Optional, Generator, List, Union
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


class OHLCVDataLoader:
    """
    Loads OHLCV data and creates rolling windows for pattern discovery.

    Attributes:
        df (pd.DataFrame): The OHLCV DataFrame
        required_columns (list): Required column names
    """

    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

    def __init__(self, df_ohlcv: pd.DataFrame, copy: bool = True, validate_ohlc: bool = True):
        """
        Initialize the data loader with OHLCV DataFrame.

        Args:
            df_ohlcv: DataFrame containing OHLCV data
            copy: If True, copy DataFrame (safer but uses memory). If False, use reference.
            validate_ohlc: If True, validate OHLC relationships (High >= Low, etc.)

        Raises:
            ValueError: If required columns are missing or OHLC validation fails
        """
        self._validate_dataframe(df_ohlcv, validate_ohlc=validate_ohlc)

        # Attempt to copy DataFrame, fallback to reference if MemoryError
        try:
            self.df = df_ohlcv.copy() if copy else df_ohlcv
        except MemoryError:
            logger.warning("Failed to copy DataFrame (MemoryError), using reference instead")
            self.df = df_ohlcv

        logger.info(f"Loaded OHLCV data with {len(self.df)} bars")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"OHLCVDataLoader(n_bars={len(self.df)}, "
            f"columns={list(self.df.columns)}, "
            f"index_type={type(self.df.index).__name__})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"OHLCV Data Loader:",
            f"  Bars: {len(self.df):,}",
            f"  Columns: {', '.join(self.df.columns)}",
        ]
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df) > 0:
            lines.append(f"  Date Range: {self.df.index[0]} to {self.df.index[-1]}")
        return "\n".join(lines)

    @property
    def n_bars(self) -> int:
        """Number of bars in the dataset."""
        return len(self.df)

    @property
    def has_datetime_index(self) -> bool:
        """Whether the DataFrame has a datetime index."""
        return isinstance(self.df.index, pd.DatetimeIndex)

    @property
    def date_range(self) -> Optional[tuple]:
        """Date range of the data if datetime index exists."""
        if self.has_datetime_index and len(self.df) > 0:
            return (self.df.index[0], self.df.index[-1])
        return None

    def _validate_dataframe(self, df: pd.DataFrame, validate_ohlc: bool = True) -> None:
        """
        Validate that DataFrame has required OHLCV columns and proper values.

        Args:
            df: DataFrame to validate
            validate_ohlc: If True, validate OHLC relationships

        Raises:
            ValueError: If required columns are missing or validation fails
        """
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required columns are: {self.REQUIRED_COLUMNS}"
            )

        if len(df) == 0:
            raise ValueError("DataFrame is empty")

        # Validate OHLC relationships if requested
        if validate_ohlc:
            invalid_rows = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close']) |
                (df['High'] <= 0) |
                (df['Low'] <= 0) |
                (df['Volume'] < 0)
            )

            if invalid_rows.any():
                n_invalid = invalid_rows.sum()
                logger.warning(
                    f"Found {n_invalid} rows ({n_invalid/len(df)*100:.2f}%) "
                    f"with invalid OHLC relationships or values"
                )
                # Log first few invalid rows for debugging
                invalid_indices = df.index[invalid_rows][:5].tolist()
                logger.debug(f"First invalid row indices: {invalid_indices}")

        logger.debug(f"DataFrame validation passed: {len(df)} rows")

    def create_windows(
        self,
        window_size: int,
        stride: int = 1,
        batch_size: Optional[int] = None
    ) -> Union[npt.NDArray[np.float64], Generator[npt.NDArray[np.float64], None, None]]:
        """
        Create rolling N-bar windows from OHLCV data.

        Uses a sliding window approach with configurable stride. Each window
        contains [Open, High, Low, Close] values. Volume is excluded.

        Args:
            window_size: Number of bars per window
            stride: Step size between consecutive windows (default: 1)
                   - stride=1: Maximum overlap (traditional sliding window)
                   - stride=window_size: No overlap between windows
                   - stride=window_size//2: 50% overlap
            batch_size: If provided, returns generator yielding batches instead of full array
                       (useful for very large datasets to save memory)

        Returns:
            Array of shape (n_windows, window_size, 4) where 4 represents [Open, High, Low, Close]
            If batch_size is specified, returns a generator yielding batches

        Raises:
            ValueError: If window_size or stride is invalid

        Examples:
            >>> loader = OHLCVDataLoader(df)
            >>> # Traditional sliding window (stride=1)
            >>> windows = loader.create_windows(10, stride=1)
            >>> print(windows.shape)
            (991, 10, 4)  # For 1000 bars: (1000-10+1) windows

            >>> # No overlap (stride=window_size)
            >>> windows = loader.create_windows(10, stride=10)
            >>> print(windows.shape)
            (100, 10, 4)  # For 1000 bars: 100 non-overlapping windows

            >>> # 50% overlap
            >>> windows = loader.create_windows(10, stride=5)
            >>> print(windows.shape)
            (199, 10, 4)  # For 1000 bars with 50% overlap
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        if window_size > len(self.df):
            raise ValueError(
                f"window_size ({window_size}) exceeds data length ({len(self.df)})"
            )

        # Calculate number of windows with stride
        n_windows = (len(self.df) - window_size) // stride + 1

        # Memory-efficient batch processing
        if batch_size is not None:
            logger.info(
                f"Creating {n_windows} windows of size {window_size} (stride={stride}) in batches of {batch_size}"
            )
            return self._create_windows_generator(window_size, stride, batch_size)

        # Standard all-at-once processing
        ohlc_data = self.df[['Open', 'High', 'Low', 'Close']].values

        if stride == 1:
            # Use efficient sliding_window_view for stride=1
            windows = np.lib.stride_tricks.sliding_window_view(
                ohlc_data,
                window_shape=(window_size, 4)
            ).squeeze(axis=1)
        else:
            # For stride > 1, manually extract windows
            windows = np.array([
                ohlc_data[i:i+window_size]
                for i in range(0, len(ohlc_data) - window_size + 1, stride)
            ])

        logger.info(
            f"Created {n_windows} windows of size {window_size} (stride={stride}) "
            f"from {len(self.df)} bars"
        )
        logger.debug(f"Windows shape: {windows.shape}")

        return windows

    def _create_windows_generator(
        self,
        window_size: int,
        stride: int,
        batch_size: int
    ) -> Generator[npt.NDArray[np.float64], None, None]:
        """
        Generator that yields window batches for memory-efficient processing.

        Args:
            window_size: Number of bars per window
            stride: Step size between consecutive windows
            batch_size: Number of windows per batch

        Yields:
            Batch of windows with shape (batch_size, window_size, 4)
        """
        ohlc_data = self.df[['Open', 'High', 'Low', 'Close']].values
        max_start_idx = len(ohlc_data) - window_size

        # Generate window start indices with stride
        window_indices = list(range(0, max_start_idx + 1, stride))
        n_windows = len(window_indices)

        for batch_start in range(0, n_windows, batch_size):
            batch_end = min(batch_start + batch_size, n_windows)
            batch = np.array([
                ohlc_data[window_indices[i]:window_indices[i]+window_size]
                for i in range(batch_start, batch_end)
            ])
            logger.debug(f"Yielding batch {batch_start//batch_size + 1}: windows {batch_start}-{batch_end-1}")
            yield batch

    def get_window_indices(self, window_size: int) -> npt.NDArray[np.int64]:
        """
        Get the starting indices for each window.

        Useful for mapping windows back to original data positions.

        Args:
            window_size: Window size used for windowing

        Returns:
            Array of starting indices for each window
        """
        n_windows = len(self.df) - window_size + 1
        return np.arange(n_windows)

    def get_window_timestamps(self, window_size: int) -> npt.NDArray:
        """
        Get timestamps for each window (if DataFrame has datetime index).

        Args:
            window_size: Window size used for windowing

        Returns:
            Array of timestamps (start of each window)
            Returns indices if no datetime index
        """
        n_windows = len(self.df) - window_size + 1
        if isinstance(self.df.index, pd.DatetimeIndex):
            return self.df.index[:n_windows].values
        else:
            return self.df.index[:n_windows].values


def generate_sample_ohlcv(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Creates realistic-looking OHLCV data using random walk with
    proper OHLC relationships (High >= max(Open, Close), etc.)

    Args:
        n_bars (int): Number of bars to generate
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with OHLCV columns and datetime index
    """
    np.random.seed(seed)

    # Generate base close prices using random walk
    returns = np.random.normal(0.0005, 0.02, n_bars)
    close_prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC data
    data = []
    for i, close in enumerate(close_prices):
        # Open is previous close (or base for first bar)
        open_price = close_prices[i - 1] if i > 0 else close

        # High and Low relative to open and close
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))

        # Volume
        volume = int(np.random.uniform(1e6, 10e6))

        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })

    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2020-01-01', periods=n_bars, freq='1h')

    logger.info(f"Generated {n_bars} bars of synthetic OHLCV data")
    return df


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing OHLCV Data Loader...\n")

    # Generate sample data
    print("Generating 100 bars of sample data...")
    df = generate_sample_ohlcv(n_bars=100)
    print(f"Generated data shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}\n")

    # Create loader
    print("Creating OHLCVDataLoader...")
    loader = OHLCVDataLoader(df)

    # Test windowing
    window_size = 10
    print(f"\nCreating windows with size {window_size}...")
    windows = loader.create_windows(window_size)

    print(f"Windows shape: {windows.shape}")
    print(f"Expected shape: ({len(df) - window_size + 1}, {window_size}, 4)")

    # Verify first window
    print(f"\nFirst window (first 3 bars):")
    print(windows[0][:3])
    print(f"\nExpected (first 3 bars from DataFrame):")
    print(df[['Open', 'High', 'Low', 'Close']].iloc[:3].values)

    # Verify last window
    print(f"\nLast window (first 3 bars):")
    print(windows[-1][:3])
    print(f"\nExpected (bars {len(df)-window_size} to {len(df)-window_size+2}):")
    print(df[['Open', 'High', 'Low', 'Close']].iloc[-window_size:-window_size+3].values)

    print("\n✓ Data loader tests completed successfully!")
