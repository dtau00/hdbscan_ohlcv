#!/usr/bin/env python3
"""Main Orchestration Script for HDBSCAN OHLCV Pattern Discovery

This script coordinates the entire hyperparameter tuning process:
1. Setup: logging, backend detection, output directory
2. Load OHLCV data
3. Generate all valid configurations
4. Main loop: for each config
   - Create windows
   - Extract features
   - Normalize features (fit scaler per window_size)
   - Run HDBSCAN
   - Compute metrics
   - Save results
   - Log progress
5. Summary: load metrics, print stats, save report
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm import tqdm

# Import project modules
from src.config import Config
from src.gpu_utils import detect_compute_backend
from src.data_loader import OHLCVDataLoader
from src.feature_engineering import FeatureExtractor
from src.clustering import HDBSCANClusterer
from src.metrics import ClusterMetrics
from src.storage import ResultsStorage


def setup_logging() -> logging.Logger:
    """
    Configure logging for the application.

    Returns:
        Logger instance for main module
    """
    # Ensure directories exist
    Config.ensure_dirs()

    # Get log file path with timestamp
    log_file = Config.get_log_file_path("hdbscan_ohlcv_main")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("HDBSCAN OHLCV Pattern Discovery - Main Pipeline")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    return logger


def load_or_generate_data(logger: logging.Logger, n_bars: int = 1000) -> pd.DataFrame:
    """
    Load OHLCV data from file or generate synthetic data for testing.

    Args:
        logger: Logger instance
        n_bars: Number of bars to generate if no data file found

    Returns:
        DataFrame with OHLCV data
    """
    logger.info("Loading OHLCV data...")

    # Check for data files in data directory
    data_files = list(Config.DATA_DIR.glob("*.csv"))

    if data_files:
        # Load the first CSV file found
        data_file = data_files[0]
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)

        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing columns in data file: {missing_cols}")
            logger.info("Generating synthetic data instead...")
            df = generate_synthetic_ohlcv(n_bars)
        else:
            logger.info(f"Loaded {len(df)} bars from {data_file}")
    else:
        logger.info(f"No data files found in {Config.DATA_DIR}")
        logger.info(f"Generating synthetic OHLCV data with {n_bars} bars...")
        df = generate_synthetic_ohlcv(n_bars)

    return df


def generate_synthetic_ohlcv(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Creates realistic-looking price data with proper OHLC relationships.

    Args:
        n_bars: Number of bars to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV columns
    """
    np.random.seed(seed)

    # Generate base price as random walk
    returns = np.random.randn(n_bars) * 0.02  # 2% volatility
    price = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC values
    close = price
    open_ = close + np.random.randn(n_bars) * 0.5

    # High and Low should respect Open and Close
    oc_max = np.maximum(open_, close)
    oc_min = np.minimum(open_, close)

    high = oc_max + np.abs(np.random.randn(n_bars)) * 0.5
    low = oc_min - np.abs(np.random.randn(n_bars)) * 0.5

    # Volume (random but realistic)
    volume = np.random.exponential(scale=1000000, size=n_bars)

    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })

    return df


def process_single_config(
    config: Dict[str, Any],
    df_ohlcv: pd.DataFrame,
    backend_type: str,
    backend_module: Any,
    storage: ResultsStorage,
    logger: logging.Logger,
    scalers_cache: Dict[int, StandardScaler]
) -> Dict[str, Any]:
    """
    Process a single configuration: window creation, feature extraction,
    clustering, metrics computation, and result storage.

    Args:
        config: Configuration dictionary
        df_ohlcv: OHLCV DataFrame
        backend_type: Compute backend type ('gpu' or 'cpu')
        backend_module: Backend module reference
        storage: ResultsStorage instance
        logger: Logger instance
        scalers_cache: Cache of fitted scalers per window_size

    Returns:
        Dictionary with run results (run_id, metrics, etc.)
    """
    window_size = config['window_size']
    config_id = Config.get_config_id(config)

    logger.info(f"Processing config: {config_id}")
    logger.info(f"  Parameters: window_size={window_size}, "
                f"min_cluster_size={config['min_cluster_size']}, "
                f"min_samples={config['min_samples']}, "
                f"metric={config['metric']}, "
                f"method={config['cluster_selection_method']}")

    try:
        # Step 1: Create windows
        logger.debug("Creating windows...")
        data_loader = OHLCVDataLoader(df_ohlcv)
        windows = data_loader.create_windows(window_size)
        logger.info(f"  Created {len(windows)} windows of size {window_size}")

        # Step 2: Extract features
        logger.debug("Extracting features...")
        feature_extractor = FeatureExtractor(feature_type='flatten', flatten_order='sequential')
        features = feature_extractor.extract_features(windows)
        logger.info(f"  Extracted features: shape={features.shape}")

        # Step 3: Normalize features
        # Use cached scaler if available for this window_size, otherwise fit new one
        if window_size not in scalers_cache:
            logger.debug(f"Fitting new StandardScaler for window_size={window_size}")
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            scalers_cache[window_size] = scaler
        else:
            logger.debug(f"Using cached StandardScaler for window_size={window_size}")
            scaler = scalers_cache[window_size]
            features_normalized = scaler.transform(features)

        logger.info(f"  Normalized features: mean={features_normalized.mean():.4f}, "
                   f"std={features_normalized.std():.4f}")

        # Step 4: Run HDBSCAN clustering
        logger.debug("Running HDBSCAN clustering...")
        clusterer_wrapper = HDBSCANClusterer(backend_type, backend_module)
        labels, clusterer = clusterer_wrapper.fit_predict(
            features_normalized,
            min_cluster_size=config['min_cluster_size'],
            min_samples=config['min_samples'],
            metric=config['metric'],
            cluster_selection_method=config['cluster_selection_method']
        )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        logger.info(f"  Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        # Step 5: Compute metrics
        logger.debug("Computing metrics...")
        metrics = ClusterMetrics.compute_metrics(labels, features_normalized)
        logger.info(f"  Metrics: n_clusters={metrics['n_clusters']}, "
                   f"noise_ratio={metrics['noise_ratio']:.3f}")

        if metrics.get('silhouette_score') is not None:
            logger.info(f"  Quality: silhouette={metrics['silhouette_score']:.4f}, "
                       f"davies_bouldin={metrics.get('davies_bouldin_score', 0):.4f}")

        # Step 6: Save results
        logger.debug("Saving results...")
        run_id = storage.save_run_results(
            config=config,
            labels=labels,
            metrics=metrics,
            clusterer=clusterer,
            features=None,  # Don't save features by default to save space
            save_clusterer=True,
            save_features=False
        )

        logger.info(f"  Results saved with run_id={run_id}")

        return {
            'run_id': run_id,
            'config': config,
            'config_id': config_id,
            'metrics': metrics,
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"  Error processing config {config_id}: {e}", exc_info=True)
        return {
            'run_id': None,
            'config': config,
            'config_id': config_id,
            'metrics': None,
            'success': False,
            'error': str(e)
        }


def print_summary_report(storage: ResultsStorage, logger: logging.Logger) -> None:
    """
    Print and save a summary report of all results.

    Args:
        storage: ResultsStorage instance
        logger: Logger instance
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*80)

    try:
        df = storage.load_metrics_dataframe()

        # Print summary statistics
        logger.info(f"\nTotal runs completed: {len(df)}")
        logger.info(f"\nConfiguration coverage:")
        logger.info(f"  Window sizes: {sorted(df['window_size'].unique())}")
        logger.info(f"  Min cluster sizes: {sorted(df['min_cluster_size'].unique())}")
        logger.info(f"  Min samples: {sorted(df['min_samples'].unique())}")
        logger.info(f"  Metrics: {sorted(df['metric'].unique())}")
        logger.info(f"  Methods: {sorted(df['cluster_selection_method'].unique())}")

        logger.info(f"\nClustering statistics:")
        logger.info(f"  Total clusters found: {df['n_clusters'].sum()}")
        logger.info(f"  Average clusters per run: {df['n_clusters'].mean():.2f} ± {df['n_clusters'].std():.2f}")
        logger.info(f"  Min clusters: {df['n_clusters'].min()}")
        logger.info(f"  Max clusters: {df['n_clusters'].max()}")
        logger.info(f"  Average noise ratio: {df['noise_ratio'].mean()*100:.2f}% ± {df['noise_ratio'].std()*100:.2f}%")

        # Quality metrics (if available)
        if 'silhouette_score' in df.columns and df['silhouette_score'].notna().any():
            logger.info(f"\nQuality metrics (average ± std):")
            logger.info(f"  Silhouette score: {df['silhouette_score'].mean():.4f} ± {df['silhouette_score'].std():.4f}")
            logger.info(f"  Davies-Bouldin score: {df['davies_bouldin_score'].mean():.4f} ± {df['davies_bouldin_score'].std():.4f}")
            logger.info(f"  Calinski-Harabasz score: {df['calinski_harabasz_score'].mean():.1f} ± {df['calinski_harabasz_score'].std():.1f}")

            # Best runs
            logger.info(f"\nTop 3 runs by Silhouette Score:")
            top_runs = df.nlargest(3, 'silhouette_score')[
                ['run_id', 'config_id', 'n_clusters', 'silhouette_score', 'noise_ratio']
            ]
            logger.info(f"\n{top_runs.to_string(index=False)}")

        # Save summary to file
        summary_file = Config.get_results_path('metrics', f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(summary_file, 'w') as f:
            f.write("HDBSCAN OHLCV Pattern Discovery - Summary Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total runs: {len(df)}\n\n")
            f.write(df.describe().to_string())
            f.write("\n\n")
            f.write("All runs:\n")
            f.write(df.to_string())

        logger.info(f"\nSummary report saved to: {summary_file}")

    except FileNotFoundError:
        logger.warning("No metrics file found - no results to summarize")
    except Exception as e:
        logger.error(f"Error generating summary report: {e}", exc_info=True)


def main():
    """
    Main orchestration function.

    Coordinates the entire hyperparameter tuning pipeline.
    """
    # Step 1: Setup
    logger = setup_logging()

    logger.info("\n" + "="*80)
    logger.info("STEP 1: SETUP")
    logger.info("="*80)

    # Detect compute backend
    logger.info("Detecting compute backend...")
    backend_type, backend_module = detect_compute_backend()
    logger.info(f"Using backend: {backend_type.upper()}")

    # Create output directories
    Config.ensure_dirs()
    logger.info(f"Output directory: {Config.RESULTS_DIR}")

    # Initialize results storage
    storage = ResultsStorage()
    logger.info(f"Next run ID will start at: {storage.run_counter}")

    # Step 2: Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOAD DATA")
    logger.info("="*80)

    # Start with small dataset for testing (can be increased)
    df_ohlcv = load_or_generate_data(logger, n_bars=1000)
    logger.info(f"Loaded OHLCV data: {len(df_ohlcv)} bars")
    logger.info(f"Data columns: {list(df_ohlcv.columns)}")
    logger.info(f"Data shape: {df_ohlcv.shape}")

    # Step 3: Generate configurations
    logger.info("\n" + "="*80)
    logger.info("STEP 3: GENERATE CONFIGURATIONS")
    logger.info("="*80)

    configs = Config.generate_hdbscan_configs()
    logger.info(f"Generated {len(configs)} valid configurations")
    Config.print_config_summary(configs)

    # Step 4: Main loop - process each configuration
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MAIN PROCESSING LOOP")
    logger.info("="*80)

    # Cache for StandardScalers (one per window_size)
    scalers_cache = {}

    # Track results
    results = []
    successful = 0
    failed = 0

    # Progress bar
    for i, config in enumerate(tqdm(configs, desc="Processing configs", unit="config"), 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"Processing configuration {i}/{len(configs)}")
        logger.info(f"{'─'*80}")

        result = process_single_config(
            config=config,
            df_ohlcv=df_ohlcv,
            backend_type=backend_type,
            backend_module=backend_module,
            storage=storage,
            logger=logger,
            scalers_cache=scalers_cache
        )

        results.append(result)

        if result['success']:
            successful += 1
        else:
            failed += 1

    # Step 5: Summary
    logger.info("\n" + "="*80)
    logger.info("STEP 5: SUMMARY")
    logger.info("="*80)

    logger.info(f"\nProcessing complete!")
    logger.info(f"  Successful runs: {successful}")
    logger.info(f"  Failed runs: {failed}")
    logger.info(f"  Total configurations: {len(configs)}")

    if failed > 0:
        logger.warning(f"\nFailed configurations:")
        for result in results:
            if not result['success']:
                logger.warning(f"  {result['config_id']}: {result['error']}")

    # Print summary report
    print_summary_report(storage, logger)

    # Final storage summary
    storage.print_summary()

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {Config.RESULTS_DIR}")
    logger.info(f"  Metrics CSV: {storage.metrics_file}")
    logger.info(f"  Labels: {Config.LABELS_DIR}")
    logger.info(f"  Models: {Config.MODELS_DIR}")
    logger.info(f"  Logs: {Config.LOGS_DIR}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
