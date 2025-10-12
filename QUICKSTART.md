# Quick Start Guide

## Launch the UI (Recommended)

```bash
# From project directory
./run_ui.sh

# Or manually
streamlit run app.py
```

Opens at: `http://localhost:8501`

## First Run Workflow

1. **Start UI** → Browser opens automatically
2. **Go to Dashboard** → See current state (empty if new)
3. **Configure & Run** → Single Run tab
4. **Adjust parameters** → Use defaults or experiment
5. **Click "Run Single Configuration"** → Wait for completion
6. **Results Explorer** → View metrics and details
7. **Visualizations** → See cluster patterns

## Quick Commands

### Run from CLI
```bash
python main.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### View Results
```bash
# All metrics
cat results/metrics/metrics.csv

# Latest log
tail -f logs/*.log | tail -1
```

### Generate Visualizations
```bash
# From Python
from tools.visualize_clusters import plot_cluster_samples
plot_cluster_samples(run_id=1, cluster_ids=[0,1], n_samples=5)
```

## Key Directories

- `data/` - Your OHLCV CSV files
- `results/metrics/` - Metrics and summaries
- `results/models/` - Saved HDBSCAN models
- `results/labels/` - Cluster assignments
- `results/visualizations/` - Generated plots
- `logs/` - Execution logs

## UI Pages

| Page | Purpose |
|------|---------|
| 🏠 Dashboard | Overview and trends |
| 🛠 Configure & Run | Start experiments |
| 📊 Results Explorer | Analyze runs |
| 📈 Visualizations | View patterns |
| 📄 Logs | Debug and monitor |
| 💾 Model Manager | Manage storage |

## Common Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| Window Size | 10-15 | Bars per pattern |
| Min Cluster Size | 10-15 | Minimum samples in cluster |
| Min Samples | 5-10 | Core point neighbors |
| Metric | euclidean | Distance measure |

## Interpreting Results

**Good Run:**
- Silhouette Score > 0.6
- Noise Ratio < 10%
- Multiple distinct clusters (2-10)

**Needs Tuning:**
- Silhouette Score < 0.4
- Noise Ratio > 20%
- Only 1 cluster or all noise

## Troubleshooting

**UI won't start:**
```bash
pip install streamlit plotly
streamlit run app.py --server.port 8502
```

**No GPU detected (optional):**
```bash
pip install cupy-cuda12x cuml-cu12
```

**Out of memory:**
- Reduce dataset size in Configure & Run
- Delete old models in Model Manager
- Uncheck "Save Features" option

## Next Steps

1. Read [docs/UI_GUIDE.md](docs/UI_GUIDE.md) for detailed features
2. Check [docs/DESIGN.md](docs/DESIGN.md) for architecture
3. Experiment with grid search for parameter tuning
4. Upload your own OHLCV data

## Support Files

- **README.md** - Full project overview
- **docs/UI_GUIDE.md** - Complete UI documentation
- **docs/DESIGN.md** - Technical architecture
- **requirements.txt** - Python dependencies

---

**You're ready to go!** 🚀

Launch the UI with `./run_ui.sh` and start exploring your OHLCV patterns.
