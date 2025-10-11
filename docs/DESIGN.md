# HDBSCAN OHLCV Pattern Discovery - Design Document

## Overview
A Python application to discover fundamental price patterns in OHLCV trading data using GPU-accelerated HDBSCAN clustering with CPU fallback. The system performs automated hyperparameter tuning, metric collection, and provides tools for cluster interpretation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                  (Orchestration Layer)                       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Data Module  │───▶│ Feature Module   │───▶│ HDBSCAN     │
│              │    │                  │    │ Module      │
└──────────────┘    └──────────────────┘    └─────────────┘
                                                    │
                              ┌─────────────────────┤
                              ▼                     ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │ Metrics Module   │    │ Visualization   │
                    │                  │    │ Module          │
                    └──────────────────┘    └─────────────────┘
```

---

## Implementation Steps

### **Step 1: Project Setup & GPU Detection**
**File:** `src/gpu_utils.py`

**Purpose:** Detect GPU availability and configure compute backend

**Implementation:**
```python
import logging

def detect_compute_backend():
    """
    Detects available compute backend (GPU/CPU)
    Returns: tuple (backend_type: str, backend_module)
    """
    - Try import cupy, check cuda.is_available()
    - Try import cuml.cluster.HDBSCAN
    - If both successful: return ('gpu', cuml_module)
    - Else: fallback to ('cpu', hdbscan_module)
    - Log the selected backend
```

**Testing:**
- Run on machine with/without GPU
- Verify correct backend selection
- Check logging output

---

### **Step 2: Data Loading & Windowing**
**File:** `src/data_loader.py`

**Purpose:** Load OHLCV data and create rolling windows

**Implementation:**
```python
class OHLCVDataLoader:
    def __init__(self, df_ohlcv: pd.DataFrame):
        - Store DataFrame reference
        - Validate columns: ['Open', 'High', 'Low', 'Close', 'Volume']

    def create_windows(self, window_size: int) -> np.ndarray:
        """
        Create rolling N-bar windows
        Returns: array of shape (n_windows, window_size, 4)
                 Note: excludes Volume column
        """
        - Use sliding window (stride=1)
        - Extract [Open, High, Low, Close] only
        - Return 3D array
```

**Testing:**
- Load small test dataset (100 bars)
- Create windows of size 10
- Verify shape: (91, 10, 4)
- Check first/last windows manually

---

### **Step 3: Feature Engineering**
**File:** `src/feature_engineering.py`

**Purpose:** Extract and flatten features from OHLCV windows

**Implementation:**
```python
class FeatureExtractor:
    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Flatten OHLCV data for each window
        Input: (n_windows, window_size, 4)
        Returns: (n_windows, window_size * 4)
        """
        - Flatten each window's OHLC values
        - For window_size=10: 40 features per window
        - For window_size=15: 60 features per window
        - Return 2D feature matrix
```

**Testing:**
- Use windows from Step 2
- Verify output shape
- Check feature ordering (all Opens, then Highs, etc. OR sequential)
- Validate no NaN values

---

### **Step 4: Feature Normalization**
**File:** `src/preprocessing.py`

**Purpose:** Standardize features using sklearn StandardScaler

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler

class FeatureNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit scaler on all features and transform
        """
        - Fit StandardScaler on entire feature matrix
        - Transform features
        - Store fitted scaler for later use
        - Return normalized features
```

**Testing:**
- Use features from Step 3
- Verify mean ≈ 0, std ≈ 1 for each feature
- Check output shape matches input

---

### **Step 5: HDBSCAN Wrapper**
**File:** `src/clustering.py`

**Purpose:** Unified interface for GPU/CPU HDBSCAN with fallback

**Implementation:**
```python
class HDBSCANClusterer:
    def __init__(self, backend_type: str, backend_module):
        - Store backend info
        - Set up logging

    def fit_predict(self, features, min_cluster_size, min_samples,
                    metric='euclidean', **kwargs):
        """
        Run HDBSCAN clustering
        Returns: (labels, clusterer_object)
        """
        - Convert features to appropriate type (cupy/numpy)
        - Initialize HDBSCAN with parameters
        - Fit and predict
        - Handle backend-specific differences
        - Return labels and fitted object
```

**Testing:**
- Use normalized features from Step 4
- Test with min_cluster_size=5, min_samples=6
- Verify labels array returned
- Check for noise label (-1)
- Count clusters found

---

### **Step 6: Metrics Collection**
**File:** `src/metrics.py`

**Purpose:** Calculate and store clustering quality metrics

**Implementation:**
```python
class ClusterMetrics:
    @staticmethod
    def compute_metrics(labels: np.ndarray, features: np.ndarray):
        """
        Compute clustering metrics
        Returns: dict with metrics
        """
        - n_clusters (excluding noise)
        - n_noise_points
        - noise_ratio
        - cluster_sizes (list)
        - silhouette_score (if >1 cluster, sample if needed)
        - davies_bouldin_score (if >1 cluster)
        - calinski_harabasz_score (if >1 cluster)
```

**Testing:**
- Use labels from Step 5
- Verify all metrics computed
- Check reasonable values
- Handle edge cases (all noise, single cluster)

---

### **Step 7: Configuration & Hyperparameter Grid**
**File:** `src/config.py`

**Purpose:** Define and validate hyperparameter configurations

**Implementation:**
```python
class HDBSCANConfig:
    WINDOW_SIZES = [10, 15]
    MIN_CLUSTER_SIZES = [5, 10]
    MIN_SAMPLES_OPTIONS = [6, 10]

    @staticmethod
    def generate_configs():
        """
        Generate all valid parameter combinations
        Validates: min_samples <= min_cluster_size
        Returns: list of config dicts
        """
        - Nested loops for all parameters
        - Filter invalid combinations
        - Return list of config dicts
```

**Testing:**
- Generate all configs
- Verify count: 2 * 2 * 2 = 8 configurations (after filtering)
- Check no invalid combinations

---

### **Step 8: Results Storage**
**File:** `src/storage.py`

**Purpose:** Save results to disk during execution

**Implementation:**
```python
class ResultsStorage:
    def __init__(self, output_dir: str):
        - Create output directory structure
        - Set up file paths for metrics CSV, labels, models

    def save_run_results(self, config, labels, metrics, clusterer=None):
        """
        Save results for a single HDBSCAN run
        """
        - Append metrics to CSV (with config parameters)
        - Save labels as compressed numpy array
        - Optionally save clusterer object (pickle/joblib)
        - Log save location
```

**Testing:**
- Run with dummy data
- Verify files created
- Load and verify metrics CSV
- Check labels can be reloaded

---

### **Step 9: Main Orchestration Loop**
**File:** `main.py`

**Purpose:** Coordinate the entire hyperparameter tuning process

**Implementation:**
```python
def main():
    # 1. Setup
    - Configure logging
    - Detect compute backend
    - Create output directory
    - Initialize ResultsStorage

    # 2. Load data (placeholder for now)
    - Load/generate OHLCV DataFrame

    # 3. Generate configs
    - Get all valid configurations

    # 4. Main loop
    for config in configs:
        - Create windows for window_size
        - Extract features
        - Normalize features (fit scaler per window_size)
        - Run HDBSCAN with config params
        - Compute metrics
        - Save results
        - Log progress

    # 5. Summary
    - Load all metrics
    - Print summary statistics
    - Save summary report
```

**Testing:**
- Run with small dataset (1000 bars)
- Verify all configs executed
- Check results saved
- Review logs

---

### **Step 10: Cluster Tree Visualization Tool**
**File:** `tools/visualize_tree.py`

**Purpose:** Interactive condensed tree visualization

**Implementation:**
```python
def plot_cluster_tree(run_id: str, results_dir: str):
    """
    Load saved clusterer and plot condensed tree
    """
    - Load clusterer object for specified run
    - Check if condensed_tree_ attribute available
    - Plot using clustertree or seaborn
    - Save to file and/or show interactively
```

**Testing:**
- Load results from Step 9
- Generate tree plot
- Verify visual output
- Test with different run IDs

---

### **Step 11: OHLCV Cluster Visualization Tool**
**File:** `tools/visualize_clusters.py`

**Purpose:** Visualize sample OHLCV sequences from clusters

**Implementation:**
```python
def plot_cluster_samples(run_id: str, cluster_ids: list,
                         n_samples: int = 5):
    """
    Plot sampled OHLCV windows from specified clusters
    """
    - Load labels for run
    - Load original OHLCV windows
    - For each cluster_id:
        - Filter windows assigned to cluster
        - Random sample n_samples
        - Create candlestick plots in grid
    - Save figure
```

**Testing:**
- Use results from Step 9
- Plot 5 samples from 2-3 clusters
- Verify candlestick charts
- Check grid layout

---

### **Step 12: Integration & End-to-End Testing**

**Testing Checklist:**
1. Full pipeline with 10k bars, all configs
2. GPU vs CPU comparison (if GPU available)
3. Verify all output files created
4. Load and review metrics CSV
5. Test both visualization tools
6. Check error handling (bad data, no clusters, etc.)
7. Verify logging completeness

---

## Directory Structure

```
hdbscan_ohlcv/
├── main.py
├── src/
│   ├── __init__.py
│   ├── gpu_utils.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── metrics.py
│   ├── config.py
│   └── storage.py
├── tools/
│   ├── visualize_tree.py
│   └── visualize_clusters.py
├── data/
│   └── (OHLCV data files)
├── results/
│   ├── metrics/
│   ├── labels/
│   └── models/
├── requirements.txt
└── DESIGN.md
```

---

## Dependencies

**Core:**
- pandas
- numpy
- scikit-learn
- hdbscan (CPU fallback)

**GPU (optional):**
- cupy
- cuml

**Visualization:**
- matplotlib
- seaborn
- mplfinance (for candlesticks)
- plotly (alternative)

**Utilities:**
- joblib (model persistence)
- tqdm (progress bars)

---

## Testing Strategy

Each step should be tested independently before proceeding:
1. **Unit tests:** Test individual functions with known inputs
2. **Integration tests:** Test module interactions
3. **Manual verification:** Visual inspection of outputs
4. **Performance tests:** Timing GPU vs CPU on same data

---

## Notes

- Start with small dataset (1k-10k bars) for initial testing
- Scale to 1M bars only after full pipeline validated
- Consider memory constraints when loading large datasets
- GPU memory may be limiting factor for very large feature matrices
- Save intermediate results frequently during long runs
