# HDBSCAN OHLCV Visualization Tools

This directory contains visualization tools for exploring HDBSCAN clustering results.

## Available Tools

### 1. visualize_tree.py

Visualizes the HDBSCAN condensed cluster tree from saved clusterer models.

**Usage:**
```bash
# View tree for a specific run
python tools/visualize_tree.py --run_id run0001

# List available runs
python tools/visualize_tree.py --list

# Save to custom location
python tools/visualize_tree.py --run_id 1 --output my_tree.png
```

**Features:**
- Shows hierarchical cluster formation
- Displays cluster persistence statistics
- Works with CPU-based HDBSCAN models
- Auto-detects and reports GPU/CPU compatibility issues

### 2. visualize_clusters.py

Visualizes sample OHLCV candlestick patterns from discovered clusters.

**Usage:**
```bash
# Visualize all clusters from run 1 with 5 samples each
python tools/visualize_clusters.py --run_id 1 --clusters all --n_samples 5

# Visualize specific clusters
python tools/visualize_clusters.py --run_id 1 --clusters 0 1 2 --n_samples 3

# Show summary only (no visualization)
python tools/visualize_clusters.py --run_id 1 --summary_only

# Use custom data file
python tools/visualize_clusters.py --run_id 1 --data_file data/my_ohlcv.csv
```

**Features:**
- Candlestick plots showing OHLC patterns
- Grid layout for comparing multiple clusters
- Random sampling with reproducible seeds
- Cluster size and distribution statistics
- Custom output paths and figure sizes

## Output Locations

By default, visualizations are saved to:
- `results/visualizations/cluster_tree.png` - Cluster trees
- `results/visualizations/clusters_run####_<config>.png` - Cluster patterns

## Requirements

Both tools require:
- matplotlib
- numpy
- pandas

The cluster visualization tool additionally uses Pillow for image handling.

## Examples

### Explore Clustering Results

```bash
# 1. First, see what runs are available
python tools/visualize_tree.py --list

# 2. View summary of clusters for a run
python tools/visualize_clusters.py --run_id 1 --summary_only

# 3. Visualize the cluster tree
python tools/visualize_tree.py --run_id 1

# 4. Visualize sample patterns from each cluster
python tools/visualize_clusters.py --run_id 1 --clusters all --n_samples 5
```

### Compare Different Runs

```bash
# Visualize clusters from multiple runs
for run_id in 1 2 3; do
    python tools/visualize_clusters.py --run_id $run_id --clusters all --n_samples 3
done
```

### High-Quality Output

```bash
# Generate larger figures with more samples
python tools/visualize_clusters.py \
    --run_id 1 \
    --clusters 0 1 2 \
    --n_samples 10 \
    --figsize 20,15 \
    --output high_res_clusters.png
```

## Notes

- **Data Consistency**: The OHLCV data used for visualization must match the data used during clustering
- **GPU Models**: Cluster tree visualization requires CPU-based models (cuML models don't support condensed trees)
- **Memory**: Large visualizations with many samples may require significant memory
- **Random Seed**: Use `--seed` parameter to ensure reproducible sample selection
