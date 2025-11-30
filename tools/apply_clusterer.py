#!/usr/bin/env python3
"""Apply a trained HDBSCAN clusterer to new data.

This script takes a saved clusterer from a previous run and applies it to
new OHLCV data to find patterns that match the discovered clusters.

Usage:
    python tools/apply_clusterer.py --run-id 1 --data data/BTCUSDT_1h_new.csv
    python tools/apply_clusterer.py --run-id 5 --data data/ETHUSDT_1d.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.storage import ResultsStorage
from src.data_loader import OHLCVDataLoader
from src.feature_engineering import FeatureExtractor
from sklearn.preprocessing import StandardScaler


def apply_clusterer_to_new_data(
    run_id: int,
    data_path: str,
    output_path: str = None,
    verbose: bool = True,
    cluster_filter: list = None
) -> pd.DataFrame:
    """
    Apply a trained clusterer to new OHLCV data.

    Args:
        run_id: Run ID of the trained clusterer to load
        data_path: Path to CSV file with new OHLCV data
        output_path: Optional path to save predictions CSV
        verbose: Print progress messages
        cluster_filter: Optional list of cluster IDs to match. Other clusters will be marked as noise.

    Returns:
        DataFrame with predictions and metadata
    """
    if verbose:
        print("="*80)
        print("HDBSCAN Pattern Matching - Apply Trained Clusterer")
        print("="*80)
        print(f"\nRun ID: {run_id}")
        print(f"Data: {data_path}")
        print()

    # Step 1: Load the trained clusterer and original configuration
    if verbose:
        print("[1/6] Loading trained clusterer...")

    storage = ResultsStorage()

    try:
        clusterer = storage.load_clusterer(run_id)
        labels_original, config = storage.load_labels(run_id)
    except FileNotFoundError as e:
        print(f"Error: Could not find saved model for run_id={run_id}")
        print(f"Details: {e}")
        sys.exit(1)

    if verbose:
        print(f"  ✓ Loaded clusterer from run {run_id}")
        print(f"  Configuration: {config}")
        print(f"  Original clusters found: {len(set(labels_original)) - (1 if -1 in labels_original else 0)}")

    # Step 2: Load new data
    if verbose:
        print(f"\n[2/6] Loading new data from {data_path}...")

    try:
        df_new = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    # Validate columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df_new.columns]

    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df_new.columns)}")
        sys.exit(1)

    if verbose:
        print(f"  ✓ Loaded {len(df_new)} bars")
        print(f"  Columns: {list(df_new.columns)}")

    # Step 3: Create windows using the same parameters
    if verbose:
        print(f"\n[3/6] Creating windows...")

    window_size = config['window_size']
    stride = config.get('stride', 1)  # Default to 1 if not in config

    loader = OHLCVDataLoader(df_new)
    windows = loader.create_windows(window_size, stride=stride)

    if verbose:
        print(f"  ✓ Created {len(windows)} windows")
        print(f"    Window size: {window_size}")
        print(f"    Stride: {stride}")

    # Step 4: Extract features using the same method
    if verbose:
        print(f"\n[4/6] Extracting features...")

    # Try to determine feature type from config or default to 'normalized'
    feature_type = config.get('feature_type', 'normalized')

    extractor = FeatureExtractor(feature_type=feature_type, flatten_order='sequential')
    features = extractor.extract_features(windows)

    if verbose:
        print(f"  ✓ Extracted features: shape={features.shape}")
        print(f"    Feature type: {feature_type}")

    # Step 5: Normalize features (if needed, matching training)
    if verbose:
        print(f"\n[5/6] Normalizing features...")

    if feature_type == 'flatten':
        # For flatten type, we need to standardize
        # Note: In production, you should save and load the scaler from training
        # For now, we fit a new scaler on the new data (approximation)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        if verbose:
            print(f"  ✓ Applied StandardScaler")
            print(f"    Mean: {features_normalized.mean():.4f}, Std: {features_normalized.std():.4f}")
    else:
        # For normalized/returns, features are already scaled
        features_normalized = features
        if verbose:
            print(f"  ✓ Features already scaled (type={feature_type})")

    # Step 6: Predict cluster labels
    if verbose:
        print(f"\n[6/6] Predicting cluster labels...")

    # Use approximate_predict for soft clustering
    predicted_labels, strengths = hdbscan_approximate_predict(clusterer, features_normalized)

    # Apply cluster filter if specified
    if cluster_filter:
        if verbose:
            print(f"  Filtering for clusters: {cluster_filter}")

        # Convert non-selected clusters to noise (-1)
        original_labels = predicted_labels.copy()
        for i, label in enumerate(predicted_labels):
            if label != -1 and label not in cluster_filter:
                predicted_labels[i] = -1
                strengths[i] = 0.0

        n_filtered = np.sum(original_labels != predicted_labels)
        if verbose:
            print(f"  Filtered out {n_filtered} matches from non-selected clusters")

    n_clusters_found = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
    n_noise = np.sum(predicted_labels == -1)

    if verbose:
        print(f"  ✓ Predictions complete!")
        print(f"    Clusters matched: {n_clusters_found}")
        print(f"    Noise points: {n_noise} ({n_noise/len(predicted_labels)*100:.1f}%)")
        print()

    # Create results DataFrame
    results_df = create_results_dataframe(
        df_new, windows, predicted_labels, strengths, window_size, stride
    )

    if verbose:
        print("="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"\nTotal windows analyzed: {len(predicted_labels)}")
        print(f"Cluster distribution:")
        for label in sorted(set(predicted_labels)):
            count = np.sum(predicted_labels == label)
            pct = count / len(predicted_labels) * 100
            label_name = "Noise" if label == -1 else f"Cluster {label}"
            print(f"  {label_name:12s}: {count:5d} ({pct:5.1f}%)")

        if n_clusters_found > 0:
            print(f"\nStrongest matches per cluster:")
            for label in sorted(set(predicted_labels)):
                if label == -1:
                    continue
                cluster_mask = predicted_labels == label
                cluster_strengths = strengths[cluster_mask]
                if len(cluster_strengths) > 0:
                    max_strength = cluster_strengths.max()
                    avg_strength = cluster_strengths.mean()
                    print(f"  Cluster {label}: max={max_strength:.3f}, avg={avg_strength:.3f}")

    # Save results if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n✓ Results saved to: {output_path}")

    if verbose:
        print("\n" + "="*80)

    return results_df


def hdbscan_approximate_predict(clusterer, features):
    """
    Predict cluster labels for new data using HDBSCAN approximate_predict.

    Args:
        clusterer: Trained HDBSCAN clusterer
        features: Feature array for new data

    Returns:
        labels: Predicted cluster labels
        strengths: Membership strengths (0-1, higher = stronger match)
    """
    try:
        # Try hdbscan.prediction.approximate_predict (separate module)
        import hdbscan.prediction
        labels, strengths = hdbscan.prediction.approximate_predict(clusterer, features)
        return labels, strengths
    except (AttributeError, ImportError):
        try:
            # Try clusterer.approximate_predict (method)
            labels, strengths = clusterer.approximate_predict(features)
            return labels, strengths
        except AttributeError:
            try:
                # Try clusterer.predict (older versions)
                print("  Warning: approximate_predict not available, using predict() instead")
                labels = clusterer.predict(features)
                # Create dummy strengths (1.0 for clustered, 0.0 for noise)
                strengths = np.where(labels == -1, 0.0, 1.0)
                return labels, strengths
            except AttributeError:
                # Last resort: use all_points_membership_vectors if available
                print("  Error: Clusterer does not support prediction")
                print("  Training with prediction_data=True is required for prediction")
                raise RuntimeError("Clusterer does not support prediction. Retrain with prediction_data=True")


def create_results_dataframe(df_original, windows, labels, strengths, window_size, stride):
    """
    Create a DataFrame with predictions aligned to original data.

    Args:
        df_original: Original OHLCV DataFrame
        windows: Array of windows
        labels: Predicted cluster labels
        strengths: Membership strengths
        window_size: Window size used
        stride: Stride used

    Returns:
        DataFrame with predictions and metadata
    """
    results = []

    for i, (window, label, strength) in enumerate(zip(windows, labels, strengths)):
        # Calculate the bar index range for this window
        start_idx = i * stride
        end_idx = start_idx + window_size - 1

        # Get the last bar's close price (end of pattern)
        if end_idx < len(df_original):
            close_price = window[-1, 3]  # Close is column 3 in OHLC

            # Get timestamp if available
            if 'timestamp' in df_original.columns or 'Open_time' in df_original.columns:
                time_col = 'timestamp' if 'timestamp' in df_original.columns else 'Open_time'
                timestamp = df_original.iloc[end_idx][time_col]
            else:
                timestamp = end_idx

            results.append({
                'window_idx': i,
                'start_bar': start_idx,
                'end_bar': end_idx,
                'timestamp': timestamp,
                'cluster': label,
                'strength': strength,
                'close_price': close_price,
            })

    results_df = pd.DataFrame(results)

    # Add human-readable cluster name
    results_df['cluster_name'] = results_df['cluster'].apply(
        lambda x: 'Noise' if x == -1 else f'Cluster_{x}'
    )

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained HDBSCAN clusterer to new data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply run 1 clusterer to new data
  python tools/apply_clusterer.py --run-id 1 --data data/BTCUSDT_new.csv

  # Only match specific clusters (e.g., clusters 2, 5, and 7)
  python tools/apply_clusterer.py --run-id 1 --data data/BTCUSDT_new.csv --clusters "2,5,7"

  # Save predictions to CSV
  python tools/apply_clusterer.py --run-id 5 --data data/ETHUSDT.csv --output predictions.csv

  # Quiet mode (no verbose output)
  python tools/apply_clusterer.py --run-id 3 --data data/test.csv --quiet
        """
    )

    parser.add_argument(
        '--run-id',
        type=int,
        required=True,
        help='Run ID of the trained clusterer to use'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV file with new OHLCV data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Path to save predictions CSV (default: predictions_runXXXX_TIMESTAMP.csv)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        '--clusters',
        type=str,
        default=None,
        help='Comma-separated list of cluster IDs to match (e.g., "2,5,7"). If not specified, all clusters are matched.'
    )

    args = parser.parse_args()

    # Parse cluster filter
    cluster_filter = None
    if args.clusters:
        try:
            cluster_filter = [int(x.strip()) for x in args.clusters.split(',')]
        except ValueError:
            print(f"Error: Invalid cluster IDs: {args.clusters}", file=sys.stderr)
            print("Cluster IDs must be comma-separated integers (e.g., '2,5,7')", file=sys.stderr)
            return 1

    # Generate default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"predictions_run{args.run_id:04d}_{timestamp}.csv"

    try:
        results_df = apply_clusterer_to_new_data(
            run_id=args.run_id,
            data_path=args.data,
            output_path=args.output,
            verbose=not args.quiet,
            cluster_filter=cluster_filter
        )

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
