# Apply Trained Clusterer to New Data

This guide shows you how to use a trained HDBSCAN model to find patterns in new data.

## Quick Start

### Method 1: Command Line (Easiest)

```bash
# Apply run 1 to new data
python tools/apply_clusterer.py --run-id 1 --data data/BTCUSDT_new.csv

# Save predictions to custom file
python tools/apply_clusterer.py --run-id 5 --data data/ETHUSDT.csv --output my_predictions.csv
```

### Method 2: Jupyter Notebook (Most Interactive)

```bash
jupyter notebook notebooks/apply_to_new_data.ipynb
```

Then:
1. Change `RUN_ID` to your preferred model
2. Change `NEW_DATA_PATH` to your data file
3. Run all cells

### Method 3: Python Script (Most Flexible)

```python
from src.storage import ResultsStorage
from src.data_loader import OHLCVDataLoader
from src.feature_engineering import FeatureExtractor
import pandas as pd

# 1. Load your trained model
storage = ResultsStorage()
clusterer = storage.load_clusterer(run_id=1)
labels_original, config = storage.load_labels(run_id=1)

# 2. Load new data
df_new = pd.read_csv('data/BTCUSDT_new.csv')

# 3. Process exactly like training
loader = OHLCVDataLoader(df_new)
windows = loader.create_windows(
    window_size=config['window_size'],
    stride=config.get('stride', 1)
)

# 4. Extract features
extractor = FeatureExtractor(
    feature_type=config.get('feature_type', 'normalized'),
    flatten_order='sequential'
)
features = extractor.extract_features(windows)

# 5. Predict clusters
predicted_labels, strengths = clusterer.approximate_predict(features)

# 6. Analyze results
print(f"Found {len(set(predicted_labels)) - 1} clusters")
for cluster_id in set(predicted_labels):
    if cluster_id == -1:
        continue
    count = (predicted_labels == cluster_id).sum()
    print(f"Cluster {cluster_id}: {count} matches")
```

## Understanding the Output

The output includes:

### 1. Cluster Labels
- **-1**: Noise (doesn't match any known pattern)
- **0, 1, 2, ...**: Cluster ID (matches a discovered pattern)

### 2. Strength Scores
- Range: 0.0 to 1.0
- Higher = stronger match to the cluster
- Use threshold (e.g., > 0.7) to filter high-confidence matches

### Example Output CSV

```csv
window_idx,start_bar,end_bar,timestamp,cluster,strength,close_price,cluster_name
0,0,9,2025-01-01,2,0.85,42000.5,Cluster_2
1,1,10,2025-01-01,-1,0.0,42100.3,Noise
2,2,11,2025-01-01,5,0.72,42050.8,Cluster_5
...
```

## Use Cases

### 1. Pattern Recognition
Find when specific patterns appear in new data:

```python
# Get all instances of Cluster 3
cluster_3_matches = df_results[df_results['cluster'] == 3]

# Get high-confidence matches only
strong_matches = df_results[
    (df_results['cluster'] != -1) &
    (df_results['strength'] > 0.7)
]
```

### 2. Trading Signals
Create alerts when patterns appear:

```python
# Monitor for specific high-value clusters
SIGNAL_CLUSTERS = [2, 5, 7]  # Clusters you identified as predictive
CONFIDENCE_THRESHOLD = 0.75

signals = df_results[
    (df_results['cluster'].isin(SIGNAL_CLUSTERS)) &
    (df_results['strength'] > CONFIDENCE_THRESHOLD)
]

print(f"Found {len(signals)} trading signals")
```

### 3. Forward Testing
Evaluate if patterns predict future moves:

```python
# Calculate returns N bars after pattern detection
FORWARD_BARS = 5

for _, row in df_results.iterrows():
    end_bar = row['end_bar']
    if end_bar + FORWARD_BARS < len(df_new):
        current_price = df_new.iloc[end_bar]['Close']
        future_price = df_new.iloc[end_bar + FORWARD_BARS]['Close']
        forward_return = (future_price - current_price) / current_price * 100

        print(f"Cluster {row['cluster']}: {forward_return:+.2f}% return")
```

### 4. Real-Time Monitoring
Apply to the most recent data:

```python
# Take last N bars
WINDOW_SIZE = config['window_size']
recent_data = df_new.tail(WINDOW_SIZE)

# Process single window
loader = OHLCVDataLoader(recent_data)
window = loader.create_windows(WINDOW_SIZE, stride=1)
features = extractor.extract_features(window)

# Predict
label, strength = clusterer.approximate_predict(features)
print(f"Current pattern: Cluster {label[0]} (strength: {strength[0]:.3f})")
```

## Tips

### Choosing the Right Model (run_id)

Look for models with:
- **High silhouette score** (> 0.4): Well-separated clusters
- **Low noise ratio** (< 20%): Most patterns get classified
- **Reasonable cluster count** (5-20): Not too granular or too broad

```python
# Find best models
df_metrics = storage.load_metrics_dataframe()
best_runs = df_metrics[
    (df_metrics['silhouette_score'] > 0.4) &
    (df_metrics['noise_ratio'] < 0.2) &
    (df_metrics['n_clusters'].between(5, 20))
].nlargest(5, 'silhouette_score')
```

### Handling Different Timeframes

If your new data has a different timeframe than training:
- Use a model trained on similar timeframe data
- Or retrain with the new timeframe

### Data Requirements

Your new data must have:
- Same columns: `Open`, `High`, `Low`, `Close`, `Volume`
- At least `window_size` bars
- Same asset class (crypto/stocks/forex)

## Workflow Example

```bash
# 1. Find your best model
python -c "from src.storage import ResultsStorage; \
storage = ResultsStorage(); \
df = storage.load_metrics_dataframe(); \
print(df.nlargest(5, 'silhouette_score')[['run_id', 'silhouette_score', 'n_clusters']])"

# 2. Download new data (if using Binance)
# Use the Streamlit app: Configure & Run > Data Management > Download from Binance

# 3. Apply the model
python tools/apply_clusterer.py --run-id 1 --data data/BTCUSDT_new.csv

# 4. Analyze results
# Open the predictions CSV in Excel/pandas or use the notebook
```

## Troubleshooting

### "Could not find saved model"
- Check run_id exists: `ls results/models/`
- Make sure you've run the training pipeline first

### "Missing required columns"
- Ensure CSV has: `Open`, `High`, `Low`, `Close`, `Volume`
- Column names are case-sensitive

### "AttributeError: approximate_predict"
- Some HDBSCAN versions only have `predict()`
- The script automatically falls back to hard assignment

### Different number of clusters found
- This is normal! New data may not contain all pattern types
- Or may contain patterns that don't match any cluster (noise)

## Next Steps

After finding patterns:

1. **Validate patterns visually**
   - Use the notebook to plot matched patterns
   - Verify they look similar to training clusters

2. **Test predictive power**
   - Calculate forward returns by cluster
   - Identify which clusters predict bullish/bearish moves

3. **Build a strategy**
   - Create entry rules (e.g., "Buy when Cluster 5 appears with strength > 0.8")
   - Backtest on historical data
   - Forward test on new data

4. **Automate monitoring**
   - Set up cron job to check latest data
   - Send alerts when high-value patterns appear
   - Log detections for later analysis
