"""Configuration Management for HDBSCAN OHLCV Clustering

This module provides centralized configuration management for the project,
including paths, logging, data processing, and GPU settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Any


class Config:
    """Central configuration for the HDBSCAN OHLCV project."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Results subdirectories
    METRICS_DIR = RESULTS_DIR / "metrics"
    LABELS_DIR = RESULTS_DIR / "labels"
    MODELS_DIR = RESULTS_DIR / "models"
    PLOTS_DIR = RESULTS_DIR / "plots"

    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    # Data processing settings
    DEFAULT_WINDOW_SIZE = 10
    MAX_WINDOWS_IN_MEMORY = 1_000_000  # Safety limit for memory usage
    BATCH_SIZE_THRESHOLD = 100_000  # Use batch processing above this many windows

    # OHLCV validation
    VALIDATE_OHLC_RELATIONSHIPS = True
    ALLOW_NEGATIVE_PRICES = False

    # GPU settings
    MIN_GPU_MEMORY_GB = 1.0  # Minimum free GPU memory required (GB)
    FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
    GPU_MEMORY_CHECK = True

    # HDBSCAN parameter grids
    WINDOW_SIZES = [10, 15]
    MIN_CLUSTER_SIZES = [5, 10]
    MIN_SAMPLES_OPTIONS = [6, 10]
    METRICS = ["euclidean"]  # Can expand to ["euclidean", "manhattan", "cosine"]

    # Feature engineering
    NORMALIZE_FEATURES = True
    FEATURE_SCALER = "standard"  # Options: "standard", "minmax", "robust"

    # Performance settings
    N_JOBS = -1  # Use all available cores (-1 for sklearn)
    RANDOM_STATE = 42

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR,
            cls.METRICS_DIR,
            cls.LABELS_DIR,
            cls.MODELS_DIR,
            cls.PLOTS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_log_file_path(cls, name: str = "hdbscan_ohlcv") -> Path:
        """
        Get path for a log file.

        Args:
            name: Base name for the log file

        Returns:
            Path to the log file
        """
        cls.ensure_dirs()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.LOGS_DIR / f"{name}_{timestamp}.log"

    @classmethod
    def get_results_path(cls, subdir: str, filename: str) -> Path:
        """
        Get path for a results file.

        Args:
            subdir: Subdirectory name ('metrics', 'labels', 'models', 'plots')
            filename: Name of the file

        Returns:
            Path to the results file
        """
        cls.ensure_dirs()
        subdir_map = {
            'metrics': cls.METRICS_DIR,
            'labels': cls.LABELS_DIR,
            'models': cls.MODELS_DIR,
            'plots': cls.PLOTS_DIR,
        }

        if subdir not in subdir_map:
            raise ValueError(f"Invalid subdir '{subdir}'. Must be one of: {list(subdir_map.keys())}")

        return subdir_map[subdir] / filename

    @classmethod
    def generate_hdbscan_configs(cls) -> List[Dict[str, Any]]:
        """
        Generate all valid HDBSCAN hyperparameter configurations.

        Validates that min_samples <= min_cluster_size for each combination.

        Returns:
            List of configuration dictionaries
        """
        configs = []

        for window_size in cls.WINDOW_SIZES:
            for min_cluster_size in cls.MIN_CLUSTER_SIZES:
                for min_samples in cls.MIN_SAMPLES_OPTIONS:
                    for metric in cls.METRICS:
                        # Validate: min_samples should be <= min_cluster_size
                        if min_samples <= min_cluster_size:
                            config = {
                                'window_size': window_size,
                                'min_cluster_size': min_cluster_size,
                                'min_samples': min_samples,
                                'metric': metric,
                            }
                            configs.append(config)

        return configs

    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current configuration.

        Returns:
            Dictionary with configuration summary
        """
        return {
            'project_root': str(cls.PROJECT_ROOT),
            'data_dir': str(cls.DATA_DIR),
            'results_dir': str(cls.RESULTS_DIR),
            'log_level': cls.LOG_LEVEL,
            'force_cpu': cls.FORCE_CPU,
            'min_gpu_memory_gb': cls.MIN_GPU_MEMORY_GB,
            'window_sizes': cls.WINDOW_SIZES,
            'min_cluster_sizes': cls.MIN_CLUSTER_SIZES,
            'min_samples_options': cls.MIN_SAMPLES_OPTIONS,
            'n_configs': len(cls.generate_hdbscan_configs()),
        }


if __name__ == "__main__":
    # Test configuration
    import pprint

    print("=" * 70)
    print("HDBSCAN OHLCV Configuration Summary")
    print("=" * 70)

    config_summary = Config.get_config_summary()
    pprint.pprint(config_summary)

    print("\n" + "=" * 70)
    print("HDBSCAN Hyperparameter Configurations")
    print("=" * 70)

    configs = Config.generate_hdbscan_configs()
    print(f"\nTotal configurations: {len(configs)}\n")

    for i, config in enumerate(configs, 1):
        print(f"{i}. {config}")

    print("\n" + "=" * 70)
    print("Creating Directories...")
    print("=" * 70)

    Config.ensure_dirs()
    print("✓ All directories created/verified")

    print("\n" + "=" * 70)
    print("Example Paths")
    print("=" * 70)

    print(f"Log file: {Config.get_log_file_path()}")
    print(f"Metrics: {Config.get_results_path('metrics', 'results.csv')}")
    print(f"Labels: {Config.get_results_path('labels', 'run_001.npz')}")
    print(f"Models: {Config.get_results_path('models', 'clusterer_001.pkl')}")
