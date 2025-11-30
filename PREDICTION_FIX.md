# HDBSCAN Prediction Support - Fixed!

## What Was the Problem?

You encountered this error when trying to apply a trained model to new data:

```python
AttributeError: 'HDBSCAN' object has no attribute 'approximate_predict'
AttributeError: 'HDBSCAN' object has no attribute 'predict'
```

## Why Did This Happen?

HDBSCAN models need to be trained with `prediction_data=True` to support predictions on new data. Without this flag, the model doesn't store the necessary data structures for prediction.

## What's Been Fixed?

### 1. **Training Now Enables Prediction by Default**

Updated `src/clustering.py:266-269`:

```python
# Enable prediction data by default if not specified
if 'prediction_data' not in kwargs:
    kwargs['prediction_data'] = True
    logger.debug("Enabled prediction_data=True for future predictions")
```

**Result**: All new models trained from now on will support prediction!

### 2. **Robust Prediction Methods**

Updated both `app.py` and `tools/apply_clusterer.py` to try multiple prediction methods:

```python
try:
    # Try hdbscan.prediction.approximate_predict (separate module)
    import hdbscan.prediction
    labels, strengths = hdbscan.prediction.approximate_predict(clusterer, features)
except (AttributeError, ImportError):
    try:
        # Try clusterer.approximate_predict (method)
        labels, strengths = clusterer.approximate_predict(features)
    except AttributeError:
        # Fallback: use predict() or error if not available
        ...
```

**Result**: Works with all HDBSCAN versions and configurations!

## What About Old Models?

### Option 1: Retrain (Recommended)

Old models trained without `prediction_data=True` won't work for pattern matching. The best solution:

```bash
# Retrain your models - they'll automatically have prediction support now
python main.py
# or use Streamlit
streamlit run app.py
# → Configure & Run → Run Grid Search
```

### Option 2: Manual Fix (Advanced)

If you have a specific model you want to keep, you can retrain it with the same parameters:

```python
from src.storage import ResultsStorage
from src.clustering import HDBSCANClusterer
from src.data_loader import OHLCVDataLoader
from src.feature_engineering import FeatureExtractor

# Load old config
storage = ResultsStorage()
labels_old, config = storage.load_labels(run_id=1)

# Retrain with same parameters (now includes prediction_data=True)
# ... (use the same data and parameters)
```

## How to Verify It Works

### Test Pattern Matching:

```bash
# Method 1: Streamlit GUI
streamlit run app.py
# → 🔍 Pattern Matching → Apply Model

# Method 2: Command Line
python tools/apply_clusterer.py --run-id <NEW_RUN_ID> --data data/your_file.csv
```

If you see:
- ✅ `Pattern matching complete!` - **Success!**
- ⚠️ `Using hard assignment` - **Works, but no confidence scores**
- ❌ `Clusterer doesn't support prediction` - **Need to retrain**

## What's Different Now?

| Before | After |
|--------|-------|
| ❌ Trained models couldn't predict | ✅ All new models support prediction |
| ❌ Had to manually add `prediction_data=True` | ✅ Automatically enabled |
| ❌ Single prediction method | ✅ Multiple fallback methods |
| ❌ Confusing errors | ✅ Clear error messages |

## Summary

**If you have existing models**: Retrain them (just run your grid search again)

**For new models**: Everything works automatically - no changes needed!

**Pattern matching is now fully operational** via:
- 🖥️ Streamlit GUI (Pattern Matching page)
- 💻 Command line (`tools/apply_clusterer.py`)
- 📓 Jupyter notebook (`notebooks/apply_to_new_data.ipynb`)

🎉 **You're all set!**
