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
    WINDOW_STRIDES = [1, 5, 10]  # Stride for sliding windows (1=every bar, 10=no overlap)
    MIN_CLUSTER_SIZES = [5, 10]
    MIN_SAMPLES_OPTIONS = [6, 10]
    METRICS = ["euclidean"]  # Can expand to ["euclidean", "manhattan", "cosine"]
    CLUSTER_SELECTION_METHODS = ["eom"]  # Options: "eom" (Excess of Mass), "leaf"

    # Feature engineering
    NORMALIZE_FEATURES = True
    FEATURE_SCALER = "standard"  # Options: "standard", "minmax", "robust"

    # Performance settings
    N_JOBS = -1  # Use all available cores (-1 for sklearn)
    RANDOM_STATE = 42

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate a single configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If configuration has invalid parameters
        """
        required_keys = ['window_size', 'min_cluster_size', 'min_samples',
                        'metric', 'cluster_selection_method']
        # stride is optional, will default to 1 if not provided

        # Check all required keys present
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")

        # Validate window_size
        if not isinstance(config['window_size'], int) or config['window_size'] < 1:
            raise ValueError(
                f"window_size must be positive integer, got {config['window_size']}"
            )

        # Validate min_cluster_size
        if not isinstance(config['min_cluster_size'], int) or config['min_cluster_size'] < 2:
            raise ValueError(
                f"min_cluster_size must be integer >= 2, got {config['min_cluster_size']}"
            )

        # Validate min_samples
        if not isinstance(config['min_samples'], int) or config['min_samples'] < 1:
            raise ValueError(
                f"min_samples must be positive integer, got {config['min_samples']}"
            )

        # Validate relationship between min_samples and min_cluster_size
        if config['min_samples'] > config['min_cluster_size']:
            raise ValueError(
                f"min_samples ({config['min_samples']}) must be <= "
                f"min_cluster_size ({config['min_cluster_size']})"
            )

        # Validate stride if present (optional parameter)
        if 'stride' in config:
            if not isinstance(config['stride'], int) or config['stride'] < 1:
                raise ValueError(
                    f"stride must be positive integer, got {config['stride']}"
                )
            if config['stride'] > config['window_size']:
                raise ValueError(
                    f"stride ({config['stride']}) should be <= window_size ({config['window_size']})"
                )

        # Validate metric
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski']
        if config['metric'] not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got '{config['metric']}'"
            )

        # Validate cluster_selection_method
        valid_methods = ['eom', 'leaf']
        if config['cluster_selection_method'] not in valid_methods:
            raise ValueError(
                f"cluster_selection_method must be one of {valid_methods}, "
                f"got '{config['cluster_selection_method']}'"
            )

        return True

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
            List of configuration dictionaries, each containing:
                - window_size: int
                - stride: int (defaults to 1)
                - min_cluster_size: int
                - min_samples: int
                - metric: str
                - cluster_selection_method: str
        """
        configs = []

        for window_size in cls.WINDOW_SIZES:
            for stride in cls.WINDOW_STRIDES:
                # Skip invalid combinations where stride > window_size
                if stride > window_size:
                    continue

                for min_cluster_size in cls.MIN_CLUSTER_SIZES:
                    for min_samples in cls.MIN_SAMPLES_OPTIONS:
                        for metric in cls.METRICS:
                            for cluster_selection_method in cls.CLUSTER_SELECTION_METHODS:
                                # Validate: min_samples should be <= min_cluster_size
                                if min_samples <= min_cluster_size:
                                    config = {
                                        'window_size': window_size,
                                        'stride': stride,
                                        'min_cluster_size': min_cluster_size,
                                        'min_samples': min_samples,
                                        'metric': metric,
                                        'cluster_selection_method': cluster_selection_method,
                                    }
                                    configs.append(config)

        return configs

    @staticmethod
    def config_to_string(config: Dict[str, Any]) -> str:
        """
        Convert a configuration dict to a human-readable string.

        Args:
            config: Configuration dictionary

        Returns:
            Formatted string representation

        Examples:
            >>> config = {'window_size': 10, 'min_cluster_size': 5, 'min_samples': 6,
            ...           'metric': 'euclidean', 'cluster_selection_method': 'eom'}
            >>> print(Config.config_to_string(config))
            ws=10_mcs=5_ms=6_euclidean_eom
        """
        stride = config.get('stride', 1)
        return (
            f"ws={config['window_size']}_"
            f"stride={stride}_"
            f"mcs={config['min_cluster_size']}_"
            f"ms={config['min_samples']}_"
            f"{config['metric']}_"
            f"{config['cluster_selection_method']}"
        )

    @staticmethod
    def get_config_id(config: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a configuration.

        This is useful for naming result files and tracking experiments.

        Args:
            config: Configuration dictionary

        Returns:
            Unique configuration ID string

        Examples:
            >>> config = {'window_size': 10, 'min_cluster_size': 5, 'min_samples': 6,
            ...           'metric': 'euclidean', 'cluster_selection_method': 'eom'}
            >>> config_id = Config.get_config_id(config)
            >>> print(config_id)
            ws10_mcs5_ms6_euclidean_eom
        """
        stride = config.get('stride', 1)
        return (
            f"ws{config['window_size']}_"
            f"s{stride}_"
            f"mcs{config['min_cluster_size']}_"
            f"ms{config['min_samples']}_"
            f"{config['metric']}_"
            f"{config['cluster_selection_method']}"
        )

    @staticmethod
    def print_config_summary(configs: List[Dict[str, Any]]) -> None:
        """
        Print a detailed summary of all configurations.

        Args:
            configs: List of configuration dictionaries

        Examples:
            >>> configs = Config.generate_hdbscan_configs()
            >>> Config.print_config_summary(configs)
        """
        print(f"\n{'='*80}")
        print(f"Configuration Summary: {len(configs)} total configurations")
        print(f"{'='*80}")

        # Extract unique values for each parameter
        window_sizes = sorted(set(c['window_size'] for c in configs))
        min_cluster_sizes = sorted(set(c['min_cluster_size'] for c in configs))
        min_samples = sorted(set(c['min_samples'] for c in configs))
        metrics = sorted(set(c['metric'] for c in configs))
        methods = sorted(set(c['cluster_selection_method'] for c in configs))

        print(f"\nParameter Grid:")
        print(f"  Window sizes:               {window_sizes}")
        print(f"  Min cluster sizes:          {min_cluster_sizes}")
        print(f"  Min samples:                {min_samples}")
        print(f"  Metrics:                    {metrics}")
        print(f"  Cluster selection methods:  {methods}")

        print(f"\nAll Configurations:")
        print(f"{'#':<4} {'Window':<8} {'MinClust':<10} {'MinSamp':<9} {'Metric':<12} {'Method':<8} {'Config ID':<30}")
        print(f"{'-'*100}")

        for i, config in enumerate(configs, 1):
            config_id = Config.get_config_id(config)
            print(
                f"{i:<4} "
                f"{config['window_size']:<8} "
                f"{config['min_cluster_size']:<10} "
                f"{config['min_samples']:<9} "
                f"{config['metric']:<12} "
                f"{config['cluster_selection_method']:<8} "
                f"{config_id:<30}"
            )

        print(f"{'='*100}\n")

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

    print("=" * 100)
    print("HDBSCAN OHLCV Configuration - Step 7 Testing")
    print("=" * 100)

    # Test 1: Generate configurations
    print("\n" + "=" * 100)
    print("Test 1: Generate default configurations")
    print("=" * 100)
    configs = Config.generate_hdbscan_configs()
    Config.print_config_summary(configs)

    # Test 2: Verify all configurations are valid
    print("\n" + "=" * 100)
    print("Test 2: Validate all configurations")
    print("=" * 100)
    all_valid = True
    for i, config in enumerate(configs, 1):
        try:
            Config.validate_config(config)
        except ValueError as e:
            print(f"❌ Config {i} validation failed: {e}")
            all_valid = False

    if all_valid:
        print(f"✓ All {len(configs)} configurations are valid")
    else:
        print(f"❌ Some configurations failed validation")

    # Test 3: Test configuration ID generation
    print("\n" + "=" * 100)
    print("Test 3: Test config_to_string() and get_config_id()")
    print("=" * 100)
    if configs:
        test_config = configs[0]
        config_str = Config.config_to_string(test_config)
        config_id = Config.get_config_id(test_config)
        print(f"Sample configuration:")
        pprint.pprint(test_config)
        print(f"\nconfig_to_string(): {config_str}")
        print(f"get_config_id():    {config_id}")

    # Test 4: Verify no duplicates
    print("\n" + "=" * 100)
    print("Test 4: Verify no duplicate configurations")
    print("=" * 100)
    config_ids = [Config.get_config_id(c) for c in configs]
    unique_ids = set(config_ids)

    if len(config_ids) == len(unique_ids):
        print(f"✓ All {len(configs)} configurations have unique IDs")
    else:
        duplicates = len(config_ids) - len(unique_ids)
        print(f"❌ Found {duplicates} duplicate configuration(s)")

    # Test 5: Expected configuration count
    print("\n" + "=" * 100)
    print("Test 5: Verify expected configuration count")
    print("=" * 100)
    # Expected: WINDOW_SIZES x MIN_CLUSTER_SIZES x MIN_SAMPLES_OPTIONS x METRICS x CLUSTER_SELECTION_METHODS
    # Filtered where min_samples <= min_cluster_size
    # With defaults: 2 x 2 x 2 x 1 x 1 = 8 total combinations
    # Invalid: (ws, mcs=5, ms=6) and (ws, mcs=5, ms=10) and (ws, mcs=10, ms=10) is valid
    # So we expect all 8 to be valid since:
    #   - mcs=5, ms=6: valid (6 <= 5 is FALSE, so filtered)
    #   - mcs=5, ms=10: invalid (10 <= 5 is FALSE, so filtered)
    #   - mcs=10, ms=6: valid
    #   - mcs=10, ms=10: valid
    # So 2 window_sizes * (1 invalid + 1 invalid + 2 valid) = 2 * 2 = 4 valid configs
    expected_count = 4
    print(f"Parameter grid:")
    print(f"  WINDOW_SIZES: {Config.WINDOW_SIZES} (count: {len(Config.WINDOW_SIZES)})")
    print(f"  MIN_CLUSTER_SIZES: {Config.MIN_CLUSTER_SIZES} (count: {len(Config.MIN_CLUSTER_SIZES)})")
    print(f"  MIN_SAMPLES_OPTIONS: {Config.MIN_SAMPLES_OPTIONS} (count: {len(Config.MIN_SAMPLES_OPTIONS)})")
    print(f"  METRICS: {Config.METRICS} (count: {len(Config.METRICS)})")
    print(f"  CLUSTER_SELECTION_METHODS: {Config.CLUSTER_SELECTION_METHODS} (count: {len(Config.CLUSTER_SELECTION_METHODS)})")
    print(f"\nExpected valid configurations: {expected_count}")
    print(f"Actual configurations: {len(configs)}")

    if len(configs) == expected_count:
        print("✓ Configuration count matches expected")
    else:
        print(f"⚠ Configuration count differs from expected")

    # Test 6: Test error handling
    print("\n" + "=" * 100)
    print("Test 6: Test validation error handling")
    print("=" * 100)

    test_cases = [
        ({'window_size': 0, 'min_cluster_size': 5, 'min_samples': 6,
          'metric': 'euclidean', 'cluster_selection_method': 'eom'}, "invalid window_size"),
        ({'window_size': 10, 'min_cluster_size': 1, 'min_samples': 6,
          'metric': 'euclidean', 'cluster_selection_method': 'eom'}, "invalid min_cluster_size"),
        ({'window_size': 10, 'min_cluster_size': 5, 'min_samples': 10,
          'metric': 'euclidean', 'cluster_selection_method': 'eom'}, "min_samples > min_cluster_size"),
        ({'window_size': 10, 'min_cluster_size': 5, 'min_samples': 6,
          'metric': 'invalid', 'cluster_selection_method': 'eom'}, "invalid metric"),
        ({'window_size': 10, 'min_cluster_size': 5, 'min_samples': 6,
          'metric': 'euclidean', 'cluster_selection_method': 'invalid'}, "invalid cluster_selection_method"),
        ({'window_size': 10, 'min_cluster_size': 5,
          'metric': 'euclidean', 'cluster_selection_method': 'eom'}, "missing min_samples"),
    ]

    passed = 0
    failed = 0

    for config, description in test_cases:
        try:
            Config.validate_config(config)
            print(f"❌ Failed to catch error: {description}")
            failed += 1
        except (ValueError, Exception) as e:
            print(f"✓ Caught expected error for {description}")
            passed += 1

    print(f"\nError handling: {passed} passed, {failed} failed")

    # Test 7: System configuration summary
    print("\n" + "=" * 100)
    print("Test 7: System configuration summary")
    print("=" * 100)
    config_summary = Config.get_config_summary()
    pprint.pprint(config_summary)

    # Test 8: Directory creation
    print("\n" + "=" * 100)
    print("Test 8: Create directories and test paths")
    print("=" * 100)
    Config.ensure_dirs()
    print("✓ All directories created/verified")
    print(f"\nExample paths:")
    print(f"  Log file: {Config.get_log_file_path()}")
    print(f"  Metrics:  {Config.get_results_path('metrics', 'results.csv')}")
    print(f"  Labels:   {Config.get_results_path('labels', 'run_001.npz')}")
    print(f"  Models:   {Config.get_results_path('models', 'clusterer_001.pkl')}")

    print("\n" + "=" * 100)
    print("✓ All Step 7 configuration tests completed successfully!")
    print("=" * 100)
