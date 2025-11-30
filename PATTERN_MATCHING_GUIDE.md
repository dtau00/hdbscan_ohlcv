# Pattern Matching Guide

## Overview

You now have **3 ways** to apply your trained HDBSCAN models to new data and find matching patterns.

---

## 🎯 Method 1: Streamlit GUI (Easiest!)

### New Page: "🔍 Pattern Matching"

Located in the Streamlit app sidebar, this page gives you a complete workflow:

```bash
streamlit run app.py
# Then navigate to: 🔍 Pattern Matching
```

### Features:

#### Tab 1: 🎯 Apply Model
1. **Select Your Trained Model**
   - Shows top 5 models by silhouette score
   - Pick from dropdown
   - See model stats (clusters, quality metrics)

2. **Select Data to Analyze**
   - Choose from your uploaded CSV files
   - Preview data before running
   - See file size and row count

3. **Run Pattern Matching**
   - Click "🔍 Find Patterns" button
   - Real-time progress updates
   - Instant results display
   - Download results as CSV

#### Tab 2: 📊 View Results
- **Summary Metrics**: Total windows, matched patterns, noise, avg strength
- **Cluster Distribution**: Bar chart showing which clusters were found
- **Detailed Results Table**:
  - Filter by cluster
  - Filter by minimum strength
  - Sort and explore all matches
- **Download Full Results**: Export to CSV

#### Tab 3: 📈 Analysis
- **Strongest Matches by Cluster**: See top 5 for each cluster
- **Pattern Timeline**: Visualization showing when patterns appear
- **Cluster Statistics**: Max/avg strength per cluster

### Advantages:
- ✅ No coding required
- ✅ Visual interface
- ✅ Interactive charts
- ✅ Built-in data preview
- ✅ Automatic result saving
- ✅ Export capabilities

---

## 💻 Method 2: Command Line Tool

### Quick Usage:

```bash
# Basic usage
python tools/apply_clusterer.py --run-id 1 --data data/BTCUSDT_new.csv

# Custom output file
python tools/apply_clusterer.py \
    --run-id 5 \
    --data data/ETHUSDT.csv \
    --output my_predictions.csv

# Quiet mode (minimal output)
python tools/apply_clusterer.py \
    --run-id 3 \
    --data data/test.csv \
    --quiet
```

### Advantages:
- ✅ Fast for batch processing
- ✅ Can be automated (cron jobs)
- ✅ Easy to script
- ✅ No GUI needed

---

## 📓 Method 3: Jupyter Notebook

### Open the Notebook:

```bash
jupyter notebook notebooks/apply_to_new_data.ipynb
```

### What's Inside:
- **Step-by-step walkthrough** with explanations
- **Code you can modify** and experiment with
- **Visual pattern analysis** (OHLC charts, candlesticks)
- **Forward return testing** (see what happens after patterns)
- **Inline visualizations** using matplotlib

### Advantages:
- ✅ Most flexible
- ✅ Can customize everything
- ✅ Learn how it works
- ✅ Great for research

---

## 🎯 Workflow Comparison

| Task | Streamlit GUI | Command Line | Jupyter Notebook |
|------|--------------|--------------|------------------|
| **Quick pattern search** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Visual exploration** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Batch processing** | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Automation** | ⭐ | ⭐⭐⭐ | ⭐ |
| **Learning/Research** | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Customization** | ⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 📋 Example Workflows

### Workflow 1: Quick Pattern Check
**Use Case**: "Did my favorite pattern appear in today's data?"

```bash
# Streamlit GUI
streamlit run app.py
# → Pattern Matching → Apply Model → Select model & data → Find Patterns
```

### Workflow 2: Batch Analysis
**Use Case**: "Analyze 10 different data files with same model"

```bash
# Command line
for file in data/*.csv; do
    python tools/apply_clusterer.py --run-id 1 --data "$file"
done
```

### Workflow 3: Research & Development
**Use Case**: "Test if Cluster 5 predicts bullish moves"

```python
# Jupyter Notebook
# 1. Load model and apply to data
# 2. Filter for Cluster 5 matches
# 3. Calculate forward returns
# 4. Visualize results
# 5. Compute statistics
```

### Workflow 4: Real-Time Monitoring
**Use Case**: "Get alerted when patterns appear"

```bash
# Cron job (runs every hour)
0 * * * * python tools/apply_clusterer.py \
    --run-id 1 \
    --data /latest/BTCUSDT.csv \
    --quiet && \
    python alert_if_cluster_5.py
```

---

## 🎓 Understanding the Output

### Output CSV Format:

```csv
window_idx,start_bar,end_bar,timestamp,cluster,strength,close_price,cluster_name
0,0,9,2025-01-01 00:00,2,0.85,42000.5,Cluster_2
1,1,10,2025-01-01 01:00,-1,0.0,42100.3,Noise
2,2,11,2025-01-01 02:00,5,0.72,42050.8,Cluster_5
```

### Column Meanings:

- **window_idx**: Index of the window in sequence
- **start_bar**: Starting bar index in original data
- **end_bar**: Ending bar index in original data
- **timestamp**: Time of the pattern's end
- **cluster**: Cluster ID (-1 = noise, 0+ = cluster)
- **strength**: Confidence score (0-1, higher = better match)
- **close_price**: Closing price at pattern end
- **cluster_name**: Human-readable cluster label

---

## 💡 Tips & Best Practices

### 1. Choosing the Right Model
Look for models with:
- **Silhouette score > 0.4**: Well-separated clusters
- **Noise ratio < 20%**: Most patterns get classified
- **5-20 clusters**: Not too granular, not too broad

```python
# In Streamlit: Dashboard shows top models
# In CLI: Check results/metrics/metrics.csv
```

### 2. Interpreting Strength Scores
- **> 0.8**: Very strong match, high confidence
- **0.6 - 0.8**: Good match, decent confidence
- **0.4 - 0.6**: Weak match, use with caution
- **< 0.4**: Very weak, likely false positive

### 3. Handling Noise (-1 cluster)
- Noise means "doesn't match any known pattern"
- High noise is normal (HDBSCAN is conservative)
- Focus on high-strength matches, not noise reduction

### 4. Data Requirements
Your new data must have:
- Same columns: `Open`, `High`, `Low`, `Close`, `Volume`
- At least `window_size` bars
- Same timeframe as training data (recommended)
- Same asset class (crypto/stocks/forex)

---

## 🔧 Troubleshooting

### "No trained models available"
**Solution**: Train a model first in "Configure & Run"

### "Missing required columns"
**Solution**: Ensure CSV has OHLCV columns (case-sensitive)

### "Different clusters found than training"
**Solution**: This is normal! New data may not contain all patterns

### "Low strength scores"
**Solution**:
- Try a different model (higher silhouette score)
- Check if data is from same timeframe/asset
- Verify data quality

---

## 📚 Next Steps

After finding patterns:

1. **Validate Visually**
   - Use Visualizations tab to see what matched patterns look like
   - Verify they're similar to training clusters

2. **Test Predictive Power**
   - Calculate forward returns (Jupyter notebook has examples)
   - Identify which clusters predict moves

3. **Build Strategy**
   - Create entry rules (e.g., "Buy when Cluster 5 + strength > 0.8")
   - Backtest on historical data
   - Forward test on new data

4. **Automate**
   - Set up monitoring scripts
   - Create alerts for high-value patterns
   - Log detections for analysis

---

## 🚀 Quick Reference

```bash
# Streamlit GUI
streamlit run app.py

# Command line
python tools/apply_clusterer.py --run-id <RUN_ID> --data <DATA_FILE>

# Jupyter
jupyter notebook notebooks/apply_to_new_data.ipynb

# Help
python tools/apply_clusterer.py --help
```

**Documentation**:
- Full guide: `APPLY_TO_NEW_DATA.md`
- Training guide: `README.md`
- Code reference: `tools/apply_clusterer.py`
