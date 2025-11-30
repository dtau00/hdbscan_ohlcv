# Cluster Filtering Guide

## Overview

You can now **select specific clusters** to match when applying your trained models! This is perfect when only some clusters are useful for your trading strategy.

---

## 🎯 Why Filter Clusters?

After training a model, you might discover:
- **Cluster 5** predicts bullish moves 70% of the time ✅
- **Cluster 2** has no predictive power ❌
- **Cluster 7** predicts reversals ✅

**Instead of getting all clusters**, you can filter to only match clusters 5 and 7!

---

## 🖥️ Method 1: Streamlit GUI (Easiest)

### Step-by-Step:

1. **Start the app**
   ```bash
   streamlit run app.py
   ```

2. **Navigate to**: 🔍 Pattern Matching → 🎯 Apply Model

3. **Select your model** (Step 1)

4. **Select your data** (Step 2)

5. **Select clusters** (Step 3 - NEW!)
   - The app shows you which clusters the model has
   - Use the multiselect to choose specific clusters
   - Or click "✅ Select All Clusters" to match everything
   - Leave empty to match all clusters

6. **Run matching** (Step 4)
   - Click "🔍 Find Patterns"
   - Only your selected clusters will appear in results!

### Example:

```
Model has 8 clusters: [0, 1, 2, 3, 4, 5, 6, 7]

Select specific clusters to match: [5, 7]  ← Choose your profitable clusters!

🎯 Will only match clusters: [5, 7]
```

**Result**: All other clusters are ignored, results only show clusters 5 and 7!

---

## 💻 Method 2: Command Line

### Usage:

```bash
# Match only clusters 2, 5, and 7
python tools/apply_clusterer.py \
    --run-id 1 \
    --data data/BTCUSDT_new.csv \
    --clusters "2,5,7"
```

### Examples:

```bash
# Match all clusters (default behavior)
python tools/apply_clusterer.py --run-id 1 --data data/test.csv

# Match only cluster 5
python tools/apply_clusterer.py --run-id 1 --data data/test.csv --clusters "5"

# Match clusters 2, 3, and 8
python tools/apply_clusterer.py --run-id 1 --data data/test.csv --clusters "2,3,8"

# Combine with other options
python tools/apply_clusterer.py \
    --run-id 1 \
    --data data/test.csv \
    --clusters "5,7" \
    --output my_filtered_results.csv
```

---

## 📓 Method 3: Jupyter Notebook

Add cluster filtering to your notebook:

```python
from src.storage import ResultsStorage
import hdbscan.prediction

# Load model
storage = ResultsStorage()
clusterer = storage.load_clusterer(run_id=1)

# Predict on new data
predicted_labels, strengths = hdbscan.prediction.approximate_predict(
    clusterer, features_normalized
)

# Filter for specific clusters
DESIRED_CLUSTERS = [5, 7]  # Only these clusters!

# Convert non-selected clusters to noise
for i, label in enumerate(predicted_labels):
    if label != -1 and label not in DESIRED_CLUSTERS:
        predicted_labels[i] = -1
        strengths[i] = 0.0

# Now only clusters 5 and 7 will appear in results
print(f"Found {len(predicted_labels[predicted_labels != -1])} matches")
print(f"Clusters found: {set(predicted_labels[predicted_labels != -1])}")
```

---

## 🎓 Understanding the Results

### Without Filtering:

```csv
window_idx,cluster,strength
0,2,0.75
1,5,0.82
2,-1,0.0
3,7,0.91
4,2,0.68
```

### With Filtering (clusters=[5,7]):

```csv
window_idx,cluster,strength
0,-1,0.0    ← Cluster 2 converted to noise
1,5,0.82    ← Kept!
2,-1,0.0
3,7,0.91    ← Kept!
4,-1,0.0    ← Cluster 2 converted to noise
```

**Only clusters 5 and 7 remain**, all others become noise (-1)!

---

## 💡 Use Cases

### 1. **Trading Signals**
Only match clusters that historically predict profitable moves:

```bash
# Only bullish patterns (clusters 5, 7, 9)
python tools/apply_clusterer.py \
    --run-id 1 \
    --data data/BTCUSDT_live.csv \
    --clusters "5,7,9"
```

### 2. **Risk Management**
Filter out clusters associated with high volatility:

```bash
# Exclude risky clusters (only 1, 3, 6)
python tools/apply_clusterer.py \
    --run-id 1 \
    --data data/ETHUSDT.csv \
    --clusters "1,3,6"
```

### 3. **Strategy Testing**
Test different cluster combinations:

```bash
# Test conservative clusters
python tools/apply_clusterer.py --run-id 1 --data test.csv --clusters "2,3"

# Test aggressive clusters
python tools/apply_clusterer.py --run-id 1 --data test.csv --clusters "7,8,9"

# Compare results!
```

### 4. **Multi-Strategy**
Create different strategies from the same model:

```bash
# Strategy A: Trend following (clusters 5, 7)
python tools/apply_clusterer.py \
    --run-id 1 --data data.csv --clusters "5,7" \
    --output strategy_a.csv

# Strategy B: Mean reversion (clusters 2, 4)
python tools/apply_clusterer.py \
    --run-id 1 --data data.csv --clusters "2,4" \
    --output strategy_b.csv
```

---

## 🔍 Finding Which Clusters Are Useful

### Step 1: Run Without Filtering

First, match all clusters to see what you get:

```bash
python tools/apply_clusterer.py --run-id 1 --data historical_data.csv
```

### Step 2: Analyze Performance

Use the Jupyter notebook to calculate forward returns by cluster:

```python
# Load results
df_pred = pd.read_csv('predictions_run0001_20250129.csv')

# Calculate forward returns for each cluster
FORWARD_BARS = 5

for cluster_id in df_pred['cluster'].unique():
    if cluster_id == -1:
        continue

    cluster_data = df_pred[df_pred['cluster'] == cluster_id]

    # Calculate returns...
    # Identify winners and losers
```

### Step 3: Filter for Winners

Once you know which clusters work:

```bash
# Only use profitable clusters
python tools/apply_clusterer.py \
    --run-id 1 \
    --data live_data.csv \
    --clusters "5,7,9"  # Your profitable clusters
```

---

## ⚠️ Important Notes

### 1. **Cluster IDs are Model-Specific**
Cluster 5 in one model is NOT the same as cluster 5 in another model!

### 2. **Check Available Clusters First**
Use the Streamlit GUI or check training results to see which clusters exist:

```python
from src.storage import ResultsStorage
storage = ResultsStorage()
labels, config = storage.load_labels(run_id=1)
print(f"Available clusters: {sorted(set(labels[labels != -1]))}")
```

### 3. **Invalid Clusters are Ignored**
If you specify cluster 99 but the model only has clusters 0-7, cluster 99 is silently ignored.

### 4. **Empty Results**
If your filtered clusters never appear in the new data, you'll get all noise (-1) results!

---

## 🚀 Quick Reference

| Task | Command |
|------|---------|
| **Match all clusters** | `--clusters` not specified |
| **Match one cluster** | `--clusters "5"` |
| **Match multiple** | `--clusters "2,5,7,9"` |
| **See available clusters** | Check Streamlit GUI or training results |
| **Test cluster combos** | Run multiple times with different `--clusters` |

---

## 📊 Example Workflow

```bash
# 1. First pass: Get all patterns
python tools/apply_clusterer.py --run-id 1 --data historical.csv --output all_patterns.csv

# 2. Analyze in Jupyter or spreadsheet
# → Discover clusters 5 and 7 are profitable

# 3. Live trading: Only match profitable clusters
python tools/apply_clusterer.py \
    --run-id 1 \
    --data live_data.csv \
    --clusters "5,7" \
    --output live_signals.csv

# 4. Get alerts when profitable patterns appear
python alert_script.py live_signals.csv
```

Perfect for building selective, high-quality trading signals! 🎯
