#!/usr/bin/env python3
"""Streamlit UI for HDBSCAN OHLCV Pattern Discovery

This app provides a comprehensive interface for:
- Configuring and running HDBSCAN clustering experiments
- Visualizing and comparing results
- Managing models and viewing logs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime
import subprocess
import logging
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.storage import ResultsStorage
from src.data_loader import OHLCVDataLoader
from src.gpu_utils import detect_compute_backend

# Page config
st.set_page_config(
    page_title="HDBSCAN OHLCV Explorer",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'storage' not in st.session_state:
    st.session_state.storage = ResultsStorage()
if 'backend_type' not in st.session_state:
    st.session_state.backend_type, st.session_state.backend_module = detect_compute_backend()


def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.markdown("# \U0001F4CA HDBSCAN OHLCV")
    st.sidebar.markdown("### Pattern Discovery")
    st.sidebar.markdown("---")

    # Show backend info
    backend_emoji = "\U0001F4BB" if st.session_state.backend_type == 'gpu' else "\U0001F4A1"
    st.sidebar.info(f"{backend_emoji} **Backend:** {st.session_state.backend_type.upper()}")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "\U0001F3E0 Dashboard",
            "\U0001F6E0 Configure & Run",
            "\U0001F4CA Results Explorer",
            "\U0001F4C8 Visualizations",
            "\U0001F4C4 Logs",
            "\U0001F4BE Model Manager"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Project:** `{Config.PROJECT_ROOT.name}`")
    st.sidebar.markdown(f"**Results:** `{len(get_all_runs())}` runs")

    # Route to appropriate page
    if "\U0001F3E0 Dashboard" in page:
        show_dashboard()
    elif "\U0001F6E0 Configure & Run" in page:
        show_configure_run()
    elif "\U0001F4CA Results Explorer" in page:
        show_results_explorer()
    elif "\U0001F4C8 Visualizations" in page:
        show_visualizations()
    elif "\U0001F4C4 Logs" in page:
        show_logs()
    elif "\U0001F4BE Model Manager" in page:
        show_model_manager()


def get_all_runs() -> pd.DataFrame:
    """Load all runs from metrics CSV."""
    try:
        return st.session_state.storage.load_metrics_dataframe()
    except FileNotFoundError:
        return pd.DataFrame()


def show_dashboard():
    """Dashboard overview page."""
    st.markdown('<p class="main-header">\U0001F3E0 Dashboard</p>', unsafe_allow_html=True)

    df = get_all_runs()

    if df.empty:
        st.warning("\U000026A0 No runs found. Go to 'Configure & Run' to start your first experiment.")
        return

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Runs", len(df))

    with col2:
        avg_clusters = df['n_clusters'].mean()
        st.metric("Avg Clusters", f"{avg_clusters:.1f}")

    with col3:
        if 'silhouette_score' in df.columns:
            best_silhouette = df['silhouette_score'].max()
            st.metric("Best Silhouette", f"{best_silhouette:.3f}")
        else:
            st.metric("Best Silhouette", "N/A")

    with col4:
        avg_noise = df['noise_ratio'].mean() * 100
        st.metric("Avg Noise %", f"{avg_noise:.1f}%")

    st.markdown("---")

    # Recent runs
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("\U0001F4C5 Recent Runs")
        recent_df = df.nlargest(10, 'run_id')[
            ['run_id', 'config_id', 'n_clusters', 'silhouette_score', 'noise_ratio', 'timestamp']
        ].copy()
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_df, width='stretch', hide_index=True)

    with col2:
        st.subheader("\U0001F3C6 Top Performers")
        if 'silhouette_score' in df.columns and df['silhouette_score'].notna().any():
            top_df = df.nlargest(5, 'silhouette_score')[
                ['run_id', 'config_id', 'silhouette_score']
            ].copy()
            st.dataframe(top_df, width='stretch', hide_index=True)
        else:
            st.info("No quality metrics available yet.")

    # Performance over time
    st.subheader("\U0001F4C8 Performance Over Time")

    if 'silhouette_score' in df.columns and df['silhouette_score'].notna().any():
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['run_id'],
            y=df['silhouette_score'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            xaxis_title="Run ID",
            yaxis_title="Silhouette Score",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, width='stretch')

    # Configuration distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("\U0001F4CA Window Size Distribution")
        window_counts = df['window_size'].value_counts().sort_index()
        fig = px.bar(
            x=window_counts.index,
            y=window_counts.values,
            labels={'x': 'Window Size', 'y': 'Count'},
            color=window_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("\U0001F4CA Cluster Distribution")
        cluster_counts = df['n_clusters'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Number of Clusters', 'y': 'Count'},
            color=cluster_counts.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, width='stretch')


def show_configure_run():
    """Configuration and run execution page."""
    st.markdown('<p class="main-header">\U0001F6E0 Configure & Run</p>', unsafe_allow_html=True)

    st.markdown("Configure HDBSCAN clustering parameters and run experiments.")

    st.info("💡 **Tip:** For a single run, select one value for each parameter. For grid search, select multiple values.")

    tabs = st.tabs(["\U0001F4CA Grid Search", "\U0001F4C1 Data Management"])

    # Tab 1: Grid Search
    with tabs[0]:
        st.subheader("Hyperparameter Grid Search")

        st.markdown("Run multiple configurations in batch mode.")

        # Data file selection
        data_files = list(Config.DATA_DIR.glob("*.csv"))

        if data_files:
            st.markdown("### 📁 Data File Selection")

            # Get file names for display
            file_names = [f.name for f in data_files]

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{len(data_files)} data files available**")

            with col2:
                if st.button("✅ Select All", key="select_all_files"):
                    st.session_state.selected_files = file_names.copy()
                    st.rerun()

            with col3:
                if st.button("❌ Deselect All", key="deselect_all_files"):
                    st.session_state.selected_files = []
                    st.rerun()

            # Initialize session state for selected files
            if 'selected_files' not in st.session_state:
                st.session_state.selected_files = [file_names[0]] if file_names else []

            # Filter out any selected files that no longer exist
            valid_selected_files = [f for f in st.session_state.selected_files if f in file_names]

            # File selector
            selected_file_names = st.multiselect(
                "Select CSV files to process",
                file_names,
                default=valid_selected_files,
                help="Select one or more CSV files. Each file will be processed with all parameter combinations."
            )

            # Update session state
            st.session_state.selected_files = selected_file_names

            # Show selection summary
            if selected_file_names:
                st.info(f"📊 Will process **{len(selected_file_names)}** file(s)")
            else:
                st.warning("⚠️ No files selected. Please select at least one file.")

            st.markdown("---")
        else:
            st.warning("⚠️ No CSV files found in data directory. The system will generate synthetic data.")
            selected_file_names = []

        st.markdown("### ⚙️ HDBSCAN Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Window Sizes**")
            window_sizes = st.multiselect(
                "Select window sizes",
                [5, 10, 15, 20, 25, 30],
                default=[10, 15],
                help="Pattern lengths to test. Testing multiple sizes helps find the optimal time scale for your data."
            )

        with col2:
            st.markdown("**Min Cluster Sizes**")
            min_cluster_sizes = st.multiselect(
                "Select min cluster sizes",
                [5, 10, 15, 20, 25],
                default=[10],
                help="Minimum cluster sizes to try. Smaller values find more granular patterns, larger values find major patterns."
            )

        with col3:
            st.markdown("**Min Samples**")
            min_samples_options = st.multiselect(
                "Select min samples",
                [3, 5, 6, 10, 15],
                default=[6, 10],
                help="Core point thresholds to test. Only valid combinations (min_samples ≤ min_cluster_size) will be used."
            )

        metrics_grid = st.multiselect(
            "Distance Metrics",
            ["euclidean", "manhattan", "cosine"],
            default=["euclidean"],
            help="Distance measures to compare. Euclidean is standard. Try cosine for shape-based clustering."
        )

        n_bars_grid = st.number_input(
            "Number of Bars",
            100, 100000, 1000, 100,
            key="grid_n_bars",
            help="Total bars to use for all grid search runs. Same dataset will be used for all configurations."
        )

        # Parallel execution settings
        st.markdown("### \u26A1 Parallel Execution")
        col1, col2 = st.columns(2)

        with col1:
            use_parallel = st.checkbox(
                "Enable Parallel Execution",
                value=True,
                help="Run grid search configurations in parallel using multiple CPU cores for faster execution. Note: Parallel mode uses CPU-only (GPU acceleration is only available in sequential mode)."
            )

            if use_parallel and st.session_state.backend_type == 'gpu':
                st.info("ℹ️ Parallel execution will use CPU-only mode (CUDA contexts cannot be shared across processes)")

        with col2:
            if use_parallel:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                default_jobs = min(cpu_count, 24)
                n_jobs = st.number_input(
                    "Number of Parallel Jobs",
                    1, 24, default_jobs,
                    key="n_jobs",
                    help=f"Number of CPU cores to use (max 24). System has {cpu_count} cores."
                )
            else:
                n_jobs = 1

        # Calculate total configs
        total_configs = len(window_sizes) * len(min_cluster_sizes) * len(min_samples_options) * len(metrics_grid)

        # Show execution plan with job utilization
        if data_files and selected_file_names:
            total_runs = total_configs * len(selected_file_names)
            if use_parallel:
                batches = (total_runs + n_jobs - 1) // n_jobs  # Ceiling division
                st.info(f"📊 **{total_configs}** configs × **{len(selected_file_names)}** file(s) = **{total_runs}** total runs | Using **{n_jobs}** parallel jobs (~{batches} batches)")
            else:
                st.info(f"📊 **{total_configs}** configs × **{len(selected_file_names)}** file(s) = **{total_runs}** total runs (sequential)")
        else:
            if use_parallel:
                batches = (total_configs + n_jobs - 1) // n_jobs
                st.info(f"ℹ️ **{total_configs}** configurations | Using **{n_jobs}** parallel jobs (~{batches} batches)")
            else:
                st.info(f"ℹ️ **{total_configs}** configurations (sequential)")

        if st.button("▶ Run Grid Search", type="primary", width='stretch'):
            if data_files and not selected_file_names:
                st.error("⚠️ Please select at least one data file!")
            else:
                run_grid_search(
                    window_sizes=window_sizes,
                    min_cluster_sizes=min_cluster_sizes,
                    min_samples_options=min_samples_options,
                    metrics=metrics_grid,
                    n_bars=n_bars_grid,
                    selected_files=selected_file_names if data_files else None,
                    use_parallel=use_parallel,
                    n_jobs=n_jobs if use_parallel else 1
                )

    # Tab 2: Data Management
    with tabs[1]:
        st.subheader("Data Management")

        # Binance Download Section
        st.markdown("### \U0001F4E5 Download from Binance (Free)")

        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input(
                "Trading Pair",
                "BTCUSDT",
                help="Trading pair symbol (e.g., BTCUSDT, ETHUSDT, BNBUSDT). Must be in uppercase."
            ).upper()

        with col2:
            interval = st.selectbox(
                "Interval",
                ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
                index=11,  # Default to 1d
                help="Candlestick timeframe. Binance supports various intervals from 1 minute to 1 month."
            )

        # Date range selection
        col1, col2 = st.columns(2)

        with col1:
            from datetime import datetime, timedelta
            default_start = datetime.now() - timedelta(days=30)
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                help="Beginning of date range to download."
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="End of date range to download."
            )

        # Estimate number of bars
        estimated_bars = estimate_binance_bars(start_date, end_date, interval)
        if estimated_bars > 0:
            st.info(f"📊 Estimated candles: ~{estimated_bars:,} bars")
        else:
            st.warning("⚠️ Invalid date range (start date must be before end date)")

        run_clustering = st.checkbox(
            "Run Clustering After Download",
            value=True,
            help="Automatically run HDBSCAN clustering on downloaded data using default parameters."
        )

        if st.button("\U0001F4E5 Download & Save Data", type="primary", width='stretch'):
            if estimated_bars <= 0:
                st.error("Invalid date range!")
            else:
                download_from_binance(symbol, interval, start_date, end_date, run_clustering)

        st.markdown("---")

        st.markdown("**Available Data Files**")
        data_files = list(Config.DATA_DIR.glob("*.csv"))

        if data_files:
            for f in data_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f.name)
                with col2:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    st.text(f"{size_mb:.2f} MB")
                with col3:
                    if st.button("\U0001F5D1", key=f"del_{f.name}"):
                        f.unlink()
                        st.rerun()
        else:
            st.info("No data files found. The system will generate synthetic data for testing.")

        st.markdown("---")
        st.markdown("**Upload New Data**")
        uploaded_file = st.file_uploader("Upload CSV file (must have OHLCV columns)", type=['csv'])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df.head(), width='stretch')

            if st.button("Save to Data Directory"):
                save_path = Config.DATA_DIR / uploaded_file.name
                df.to_csv(save_path, index=False)
                st.success(f"Saved to {save_path}")
                st.rerun()


def estimate_binance_bars(start_date, end_date, interval: str) -> int:
    """Estimate number of bars between two dates for a given interval."""
    from datetime import datetime, timedelta

    # Convert dates to datetime if they're date objects
    if not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())

    if start_date >= end_date:
        return 0

    # Calculate time difference
    time_diff = end_date - start_date
    total_minutes = time_diff.total_seconds() / 60

    # Map interval to minutes
    interval_minutes = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
        "3d": 4320,
        "1w": 10080,
        "1M": 43200  # Approximate (30 days)
    }

    interval_mins = interval_minutes.get(interval, 1)
    estimated = int(total_minutes / interval_mins)

    return estimated


def download_from_binance(symbol: str, interval: str, start_date, end_date, run_clustering: bool = True) -> None:
    """Download data from Binance public API using date range and optionally run clustering."""
    import requests
    from datetime import datetime

    # Convert dates to datetime and then to milliseconds timestamp
    if not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.max.time())

    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    with st.spinner(f"Downloading {symbol} {interval} data from {start_date.date()} to {end_date.date()}..."):
        try:
            # Binance public API endpoint (no auth required)
            url = "https://api.binance.com/api/v3/klines"

            all_data = []
            current_start = start_ms

            # Binance limits to 1000 candles per request, so we may need multiple requests
            while current_start < end_ms:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ms,
                    'limit': 1000
                }

                response = requests.get(url, params=params)

                if response.status_code != 200:
                    st.error(f"Failed to download data: {response.json().get('msg', 'Unknown error')}")
                    return

                data_json = response.json()

                if not data_json:
                    break  # No more data

                all_data.extend(data_json)

                # Update start time to the last candle's close time + 1ms
                current_start = data_json[-1][6] + 1  # Close time is at index 6

                # If we got less than 1000 candles, we're done
                if len(data_json) < 1000:
                    break

            if not all_data:
                st.error(f"No data returned for {symbol}. Check symbol name and date range.")
                return

            # Convert to DataFrame
            # Binance klines format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
            df = pd.DataFrame(all_data, columns=[
                'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_volume', 'Trades', 'Taker_buy_base',
                'Taker_buy_quote', 'Ignore'
            ])

            # Convert timestamp to datetime
            df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
            df.set_index('Open_time', inplace=True)

            # Keep only OHLCV columns and convert to float
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = df[required_cols].copy()
            data = data.astype(float)

            # Remove any NaN rows
            data = data.dropna()

            # Save to file
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            filename = f"{symbol}_{interval}_{start_str}-{end_str}_{len(data)}bars.csv"
            save_path = Config.DATA_DIR / filename
            data.to_csv(save_path)

            st.success(f"✅ Downloaded {len(data):,} candles to {filename}")

            # Show preview
            st.write("**Data Preview:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("First 5 rows:")
                st.dataframe(data.head(), width='stretch')
            with col2:
                st.write("Last 5 rows:")
                st.dataframe(data.tail(), width='stretch')

            # Display stats
            st.write("**Data Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candles", f"{len(data):,}")
            with col2:
                st.metric("Interval", interval)
            with col3:
                st.metric("Start", data.index[0].strftime('%Y-%m-%d %H:%M'))
            with col4:
                st.metric("End", data.index[-1].strftime('%Y-%m-%d %H:%M'))

            # Run clustering if requested
            if run_clustering:
                st.markdown("---")
                st.markdown("### ▶ Running Clustering on Downloaded Data")

                from main import process_single_config
                import logging

                logger = logging.getLogger(__name__)

                # Default clustering configuration
                config = {
                    'window_size': 10,
                    'min_cluster_size': 10,
                    'min_samples': 6,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'
                }

                with st.spinner("Running HDBSCAN clustering..."):
                    result = process_single_config(
                        config=config,
                        df_ohlcv=data,
                        backend_type=st.session_state.backend_type,
                        backend_module=st.session_state.backend_module,
                        storage=st.session_state.storage,
                        logger=logger,
                        scalers_cache={}
                    )

                    if result['success']:
                        st.success(f"✅ Clustering completed! Run ID: {result['run_id']}")
                        st.json(result['metrics'])
                    else:
                        st.error(f"❌ Clustering failed: {result['error']}")

        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


def run_grid_search(window_sizes, min_cluster_sizes, min_samples_options, metrics, n_bars, selected_files=None, use_parallel=False, n_jobs=1):
    """Run grid search with multiple configurations across multiple files."""
    import time

    # Start timing
    start_time = time.time()

    # Generate configs
    configs = []
    for ws in window_sizes:
        for mcs in min_cluster_sizes:
            for ms in min_samples_options:
                for m in metrics:
                    if ms <= mcs:  # Validate constraint
                        configs.append({
                            'window_size': ws,
                            'min_cluster_size': mcs,
                            'min_samples': ms,
                            'metric': m,
                            'cluster_selection_method': 'eom'
                        })

    if not configs:
        st.error("No valid configurations generated!")
        return

    from main import process_single_config
    import logging
    import pandas as pd
    logger = logging.getLogger(__name__)

    # Determine which files to process
    if selected_files:
        # User selected specific files
        data_files = [Config.DATA_DIR / filename for filename in selected_files]
        total_runs = len(configs) * len(data_files)
        execution_mode = "parallel" if use_parallel else "sequential"
        st.info(f"Running {len(configs)} configurations on {len(data_files)} files = {total_runs} total runs ({execution_mode}, {n_jobs} jobs)...")
    else:
        # No files selected, use synthetic data once
        data_files = [None]  # Sentinel for synthetic data
        total_runs = len(configs)
        execution_mode = "parallel" if use_parallel else "sequential"
        st.info(f"Running {len(configs)} configurations ({execution_mode}, {n_jobs} jobs)...")

    all_results = []

    if use_parallel:
        # Parallel execution
        from src.parallel_grid_search import parallel_multi_file_grid_search

        # Prepare data files
        data_file_tuples = []
        for data_file in data_files:
            if data_file is None:
                from main import load_or_generate_data
                df_ohlcv = load_or_generate_data(logger, n_bars=n_bars)
                file_display = "Synthetic Data"
            else:
                try:
                    df_ohlcv = pd.read_csv(data_file)
                    file_display = data_file.name

                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df_ohlcv.columns for col in required_cols):
                        st.warning(f"⚠️ Skipping {file_display}: Missing required columns")
                        continue
                except Exception as e:
                    st.warning(f"⚠️ Error loading {data_file.name}: {e}")
                    continue

            data_file_tuples.append((df_ohlcv, file_display))

        # Create status display containers
        st.markdown("### ⚡ Parallel Execution Status")

        # Create a status file for tracking progress
        import tempfile
        import json
        status_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
        status_file_path = status_file.name
        status_file.close()

        # Initialize status
        initial_status = {
            'total': total_runs,
            'completed': 0,
            'running': {},  # Dict: {config_id: {config details, pid, start_time}}
            'completed_list': [],
            'failed': []
        }
        with open(status_file_path, 'w') as f:
            json.dump(initial_status, f)

        # Create status display containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        running_container = st.empty()

        status_text.info(f"⚡ Starting parallel execution with {n_jobs} workers...")

        # Capture session state values before thread starts
        # (session state is not accessible from background threads)
        backend_type = st.session_state.backend_type
        backend_module = st.session_state.backend_module
        storage = st.session_state.storage

        # Start parallel grid search in background
        import threading
        results_container: Dict[str, Any] = {'results': None, 'error': None}

        def run_parallel():
            try:
                results = parallel_multi_file_grid_search(
                    configs=configs,
                    data_files=data_file_tuples,
                    backend_type=backend_type,
                    backend_module=backend_module,
                    storage=storage,
                    process_func=process_single_config,
                    n_jobs=n_jobs,
                    verbose=10,
                    status_file=status_file_path
                )
                results_container['results'] = results
            except Exception as e:
                results_container['error'] = e

        thread = threading.Thread(target=run_parallel)
        thread.start()

        # Monitor progress while thread runs
        try:
            while thread.is_alive():
                try:
                    with open(status_file_path, 'r') as f:
                        current_status = json.load(f)

                    completed = current_status['completed']
                    total = current_status['total']
                    running_jobs = current_status.get('running', {})

                    # Update progress bar
                    progress = completed / total if total > 0 else 0
                    progress_bar.progress(progress)

                    # Update status text
                    status_text.info(f"⚡ Progress: {completed}/{total} completed | {len(running_jobs)} running")

                    # Show currently running jobs
                    if running_jobs:
                        running_text = "**Currently Running:**\n\n"
                        for config_id, job_info in list(running_jobs.items())[:5]:  # Show max 5
                            ws = job_info.get('window_size', '?')
                            mcs = job_info.get('min_cluster_size', '?')
                            ms = job_info.get('min_samples', '?')
                            metric = job_info.get('metric', '?')
                            file = job_info.get('file', 'N/A')
                            running_text += f"- `{config_id}` (ws={ws}, mcs={mcs}, ms={ms}, metric={metric}) on {file}\n"

                        if len(running_jobs) > 5:
                            running_text += f"\n...and {len(running_jobs) - 5} more"

                        running_container.markdown(running_text)

                except Exception:
                    pass  # Ignore status read errors

                # Sleep briefly before next check
                import time
                time.sleep(0.5)

            # Wait for thread to complete
            thread.join()

            # Check if error occurred
            if results_container['error']:
                raise results_container['error']

            all_results = results_container['results']

            # After completion, read final status and display summary
            try:
                with open(status_file_path, 'r') as f:
                    final_status = json.load(f)

                progress_bar.progress(1.0)
                status_text.success(f"✅ Parallel execution complete: {final_status['completed']}/{final_status['total']} configs processed")

                # Show final summary in an expander
                with st.expander("📊 Execution Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Configs", final_status['total'])
                    with col2:
                        st.metric("Completed", final_status['completed'])
                    with col3:
                        st.metric("Failed", len(final_status['failed']))

                    # Show completed list if available
                    if final_status.get('completed_list'):
                        st.markdown("### ✅ All Completed Jobs")
                        completed_data = []
                        for item in final_status['completed_list']:
                            completed_data.append({
                                'Config ID': item['config_id'],
                                'Window': item.get('window_size', '?'),
                                'Min Cluster': item.get('min_cluster_size', '?'),
                                'Min Samples': item.get('min_samples', '?'),
                                'Metric': item.get('metric', '?'),
                                'File': item.get('file', 'N/A'),
                                'Status': '✅ Success' if item.get('success') else '❌ Failed'
                            })

                        completed_df = pd.DataFrame(completed_data)
                        st.dataframe(completed_df, use_container_width=True, hide_index=True)

                    # Show failed jobs if any
                    if final_status['failed']:
                        st.markdown("### ❌ Failed Jobs")
                        failed_data = []
                        for item in final_status['failed']:
                            failed_data.append({
                                'Config ID': item['config_id'],
                                'Window': item.get('window_size', '?'),
                                'Min Cluster': item.get('min_cluster_size', '?'),
                                'Min Samples': item.get('min_samples', '?'),
                                'Metric': item.get('metric', '?'),
                                'File': item.get('file', 'N/A'),
                                'Error': item.get('error', 'Unknown error')[:80]
                            })

                        failed_df = pd.DataFrame(failed_data)
                        st.dataframe(failed_df, use_container_width=True, hide_index=True)

            except Exception as e:
                status_text.warning(f"Could not read final status: {e}")

        finally:
            # Clean up status file
            import os
            try:
                os.unlink(status_file_path)
            except:
                pass
    else:
        # Sequential execution with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        run_counter = 0

        for file_idx, data_file in enumerate(data_files):
            # Load data for this file
            if data_file is None:
                from main import load_or_generate_data
                df_ohlcv = load_or_generate_data(logger, n_bars=n_bars)
                file_display = "Synthetic Data"
            else:
                try:
                    df_ohlcv = pd.read_csv(data_file)
                    file_display = data_file.name

                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df_ohlcv.columns for col in required_cols):
                        st.warning(f"⚠️ Skipping {file_display}: Missing required columns")
                        continue

                except Exception as e:
                    st.warning(f"⚠️ Error loading {data_file.name}: {e}")
                    continue

            scalers_cache = {}

            for config_idx, config in enumerate(configs):
                run_counter += 1
                config_id = Config.get_config_id(config)

                if len(data_files) > 1:
                    status_text.text(f"File {file_idx+1}/{len(data_files)} ({file_display}) | Config {config_idx+1}/{len(configs)}: {config_id}")
                else:
                    status_text.text(f"Processing {config_idx+1}/{len(configs)}: {config_id}")

                result = process_single_config(
                    config=config,
                    df_ohlcv=df_ohlcv,
                    backend_type=st.session_state.backend_type,
                    backend_module=st.session_state.backend_module,
                    storage=st.session_state.storage,
                    logger=logger,
                    scalers_cache=scalers_cache
                )

                result['data_file'] = file_display
                all_results.append(result)

                progress_bar.progress(run_counter / total_runs)

        status_text.empty()
        progress_bar.empty()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    successful = sum(1 for r in all_results if r['success'])

    # Format time string
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.1f}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"

    if len(data_files) > 1:
        st.success(f"✅ Grid search complete! {successful}/{total_runs} runs successful across {len(data_files)} files")
    else:
        st.success(f"✅ Grid search complete! {successful}/{len(configs)} runs successful")

    st.info(f"⏱️ Total execution time: **{time_str}**")

    # Show per-file summary if multiple files
    if selected_files and len(selected_files) > 1:
        st.markdown("### 📊 Per-File Summary")
        file_summary = {}
        for result in all_results:
            file_name = result.get('data_file', 'Unknown')
            if file_name not in file_summary:
                file_summary[file_name] = {'total': 0, 'success': 0}
            file_summary[file_name]['total'] += 1
            if result['success']:
                file_summary[file_name]['success'] += 1

        summary_df = pd.DataFrame([
            {'File': k, 'Successful': v['success'], 'Total': v['total'], 'Success Rate': f"{v['success']/v['total']*100:.1f}%"}
            for k, v in file_summary.items()
        ])
        st.dataframe(summary_df, width='stretch', hide_index=True)


def show_results_explorer():
    """Results exploration and comparison page."""
    st.markdown('<p class="main-header">\U0001F4CA Results Explorer</p>', unsafe_allow_html=True)

    df = get_all_runs()

    if df.empty:
        st.warning("\U000026A0 No results available yet.")
        return

    # Filters
    st.subheader("\U0001F50D Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        window_filter = st.multiselect(
            "Window Size",
            sorted(df['window_size'].unique()),
            default=sorted(df['window_size'].unique()),
            help="Filter runs by pattern window size. Select multiple to compare different time scales."
        )

    with col2:
        min_clusters = st.number_input(
            "Min Clusters",
            0, int(df['n_clusters'].max()), 0,
            help="Only show runs that found at least this many clusters. Use to filter out poor clustering results."
        )

    with col3:
        max_noise = st.slider(
            "Max Noise Ratio",
            0.0, 1.0, 1.0, 0.05,
            help="Maximum acceptable noise ratio. Lower values show only clean clustering results. Typical threshold: 0.1-0.2."
        )

    with col4:
        sort_by = st.selectbox(
            "Sort By",
            ['run_id', 'silhouette_score', 'n_clusters', 'noise_ratio'],
            help="Sort results by: Run ID (chronological), Silhouette (quality), Clusters (count), or Noise (ratio)."
        )

    # Apply filters
    filtered_df = df[
        (df['window_size'].isin(window_filter)) &
        (df['n_clusters'] >= min_clusters) &
        (df['noise_ratio'] <= max_noise)
    ].sort_values(by=sort_by, ascending=False)

    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} runs**")

    # Results table
    st.dataframe(
        filtered_df[[
            'run_id', 'config_id', 'window_size', 'min_cluster_size', 'min_samples',
            'n_clusters', 'silhouette_score', 'davies_bouldin_score', 'noise_ratio'
        ]],
        width='stretch',
        hide_index=True,
        height=400
    )

    # Detailed view
    st.markdown("---")
    st.subheader("\U0001F50E Detailed Run View")

    run_id = st.selectbox("Select Run ID", sorted(df['run_id'].unique(), reverse=True))

    if run_id:
        run_data = df[df['run_id'] == run_id].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Configuration**")
            st.write(f"Config ID: `{run_data['config_id']}`")
            st.write(f"Window Size: {run_data['window_size']}")
            st.write(f"Min Cluster Size: {run_data['min_cluster_size']}")
            st.write(f"Min Samples: {run_data['min_samples']}")
            st.write(f"Metric: {run_data['metric']}")

        with col2:
            st.markdown("**Clustering Results**")
            st.write(f"Clusters Found: {run_data['n_clusters']}")
            st.write(f"Noise Points: {run_data['n_noise_points']}")
            st.write(f"Noise Ratio: {run_data['noise_ratio']:.1%}")
            if pd.notna(run_data.get('mean_cluster_size')):
                st.write(f"Avg Cluster Size: {run_data['mean_cluster_size']:.1f}")

        with col3:
            st.markdown("**Quality Metrics**")
            if pd.notna(run_data.get('silhouette_score')):
                st.write(f"Silhouette: {run_data['silhouette_score']:.4f}")
                st.write(f"Davies-Bouldin: {run_data['davies_bouldin_score']:.4f}")
                st.write(f"Calinski-Harabasz: {run_data['calinski_harabasz_score']:.1f}")
            else:
                st.write("No quality metrics available")


def show_visualizations():
    """Visualization page for clusters and trees."""
    st.markdown('<p class="main-header">\U0001F4C8 Visualizations</p>', unsafe_allow_html=True)

    df = get_all_runs()

    if df.empty:
        st.warning("\U000026A0 No results available yet.")
        return

    tabs = st.tabs(["\U0001F4CA Cluster Comparison", "\U0001F333 Dendrogram", "\U0001F4C9 OHLCV Patterns"])

    # Tab 1: Cluster Comparison
    with tabs[0]:
        st.subheader("Multi-Run Comparison")

        # Scatter plot: Silhouette vs Noise Ratio
        if 'silhouette_score' in df.columns:
            fig = px.scatter(
                df,
                x='noise_ratio',
                y='silhouette_score',
                size='n_clusters',
                color='window_size',
                hover_data=['run_id', 'config_id', 'min_cluster_size'],
                labels={
                    'noise_ratio': 'Noise Ratio',
                    'silhouette_score': 'Silhouette Score',
                    'n_clusters': 'Clusters',
                    'window_size': 'Window Size'
                },
                title="Quality vs Noise Trade-off"
            )
            st.plotly_chart(fig, width='stretch')

        # Parallel coordinates plot
        st.subheader("Parameter Space Exploration")
        if len(df) > 0 and 'silhouette_score' in df.columns:
            plot_df = df[['window_size', 'min_cluster_size', 'min_samples',
                         'n_clusters', 'silhouette_score']].copy()

            fig = px.parallel_coordinates(
                plot_df,
                color='silhouette_score',
                dimensions=['window_size', 'min_cluster_size', 'min_samples', 'n_clusters'],
                color_continuous_scale='Viridis',
                title="Hyperparameter Exploration"
            )
            st.plotly_chart(fig, width='stretch')

    # Tab 2: Dendrogram
    with tabs[1]:
        st.subheader("Cluster Hierarchy Visualization")

        run_id = st.selectbox(
            "Select Run ID for Tree View",
            sorted(df['run_id'].unique(), reverse=True),
            key="tree_run",
            help="Choose a run to visualize its hierarchical cluster tree (condensed dendrogram)."
        )

        if run_id is None:
            st.info("Please select a run ID to visualize.")
        elif st.button(
            "Generate Dendrogram",
            help="Creates a visualization showing how clusters form at different density levels."
        ):
            try:
                from tools.visualize_tree import ClusterTreeVisualizer

                with st.spinner("Generating dendrogram..."):
                    visualizer = ClusterTreeVisualizer(results_dir=str(Config.RESULTS_DIR))

                    # Convert run_id (int) to string format for loading
                    run_id_str = f"run{int(run_id):04d}"

                    # Load clusterer
                    clusterer, clusterer_path = visualizer.load_clusterer(run_id_str)

                    # Generate output path
                    output_path = Config.RESULTS_DIR / "visualizations" / f"tree_run{int(run_id):04d}.png"

                    # Plot tree
                    result_path = visualizer.plot_tree(
                        clusterer,
                        output_path=str(output_path),
                        show=False
                    )

                    st.success(f"Dendrogram saved to {result_path}")
                    st.image(str(result_path))
            except Exception as e:
                st.error(f"Error generating dendrogram: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    # Tab 3: OHLCV Patterns
    with tabs[2]:
        st.subheader("OHLCV Cluster Patterns")

        run_id = st.selectbox(
            "Select Run ID",
            sorted(df['run_id'].unique(), reverse=True),
            key="patterns_run",
            help="Select which clustering run to visualize."
        )

        if run_id is None:
            st.info("Please select a run ID to visualize.")
        else:
            # Check for cached visualizations
            from pathlib import Path
            results_dir_name = f"run{int(run_id):04d}"
            cache_dir = Path('.cache/viz') / results_dir_name

            cache_exists = cache_dir.exists() and (cache_dir / "metadata.pkl").exists()

            if cache_exists:
                import pickle
                with open(cache_dir / "metadata.pkl", 'rb') as f:
                    cache_metadata = pickle.load(f)

                st.success(f"✅ Pre-generated cache found ({cache_metadata['n_clusters']} clusters, {cache_metadata['n_samples']} samples each)")
                st.info(f"📁 Cache location: `{cache_dir}`")

                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🗑️ Clear Cache", help="Delete pre-generated images to regenerate with different settings"):
                        import shutil
                        shutil.rmtree(cache_dir)
                        st.success("Cache cleared!")
                        st.rerun()
                with col2:
                    st.info(f"Generated: {cache_metadata.get('generated_at', 'Unknown')}")

                st.markdown("---")
            else:
                st.info("ℹ️ No pre-generated cache found. Use pre-generation for faster visualization.")
            # Get actual cluster info from labels file (more reliable than metrics CSV)
            try:
                from src.storage import ResultsStorage
                import numpy as np
                temp_storage = ResultsStorage()
                labels, config = temp_storage.load_labels(int(run_id))
                unique_labels = np.unique(labels)
                actual_cluster_ids = sorted([int(x) for x in unique_labels if x != -1])
                n_clusters = len(actual_cluster_ids)
            except Exception as e:
                st.error(f"Error loading cluster data: {e}")
                return

            if n_clusters == 0:
                st.warning("⚠️ This run has no clusters (all points classified as noise).")

            else:
                # Pre-generation section
                st.markdown("### ⚡ Visualizations")

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    pregen_samples = st.number_input(
                        "Samples per Cluster (for pre-generation)",
                        1, 50, 10,
                        key="pregen_samples",
                        help="Number of samples to generate for each cluster"
                    )

                with col2:
                    import multiprocessing
                    detected_cores = multiprocessing.cpu_count()
                    pregen_jobs = st.number_input(
                        "CPU Cores",
                        1, 32, min(detected_cores, 20),
                        key="pregen_jobs",
                        help=f"Number of CPU cores to use. Detected: {detected_cores} cores. Recommended: use all available cores for fastest generation."
                    )

                with col3:
                    if st.button("Visualize Clusters", type="primary", help="Run standalone script with true parallelization"):
                        # Find the matching data file
                        from src.storage import ResultsStorage
                        temp_storage = ResultsStorage()
                        labels_data, config_data = temp_storage.load_labels(int(run_id))
                        expected_windows = len(labels_data)
                        window_size_data = config_data['window_size']
                        expected_bars_data = expected_windows + window_size_data - 1

                        # Find best matching data file
                        data_files_search = list(Config.DATA_DIR.glob("*.csv"))
                        best_match_file = None
                        min_diff_val = float('inf')

                        if data_files_search:
                            for data_file_item in data_files_search:
                                try:
                                    df_temp_check = pd.read_csv(data_file_item)
                                    n_bars_check = len(df_temp_check)
                                    diff_val = abs(n_bars_check - expected_bars_data)
                                    if diff_val < min_diff_val:
                                        min_diff_val = diff_val
                                        best_match_file = data_file_item
                                except:
                                    continue

                        if best_match_file:
                            # Save windows.npy for the script
                            from src.data_loader import OHLCVDataLoader
                            import numpy as np_save
                            ohlcv_data_pregen = pd.read_csv(best_match_file)
                            loader = OHLCVDataLoader(ohlcv_data_pregen, copy=False)
                            windows_result = loader.create_windows(window_size_data)

                            # Ensure it's an array, not a generator
                            if not isinstance(windows_result, np_save.ndarray):
                                windows_array = np_save.array(list(windows_result))
                            else:
                                windows_array = windows_result

                            # Save to results directory for this run
                            results_run_dir = Config.RESULTS_DIR / "labels" / f"run{int(run_id):04d}"
                            results_run_dir.mkdir(parents=True, exist_ok=True)
                            windows_save_path = results_run_dir / "windows.npy"

                            np_save.save(windows_save_path, windows_array)

                            # Find results.pkl (use labels file as results)
                            labels_pattern_search = f"labels_run{int(run_id):04d}_*.npz"
                            labels_files = list(Config.LABELS_DIR.glob(labels_pattern_search))

                            if labels_files:
                                # Create a temporary results.pkl with labels
                                results_pkl_path = results_run_dir / "results.pkl"
                                import pickle as pkl_save
                                with open(results_pkl_path, 'wb') as f_save:
                                    pkl_save.dump({'labels': labels_data}, f_save)

                                # Run pre-generation script
                                import subprocess as subp
                                import sys
                                cmd = [
                                    sys.executable,  # Use same Python interpreter as Streamlit
                                    "tools/pregenerate_viz.py",
                                    "--results", str(results_pkl_path),
                                    "--samples", str(pregen_samples),
                                    "--jobs", str(pregen_jobs)
                                ]

                                with st.spinner(f"Pre-generating {n_clusters} clusters with {pregen_samples} samples each using {pregen_jobs} cores..."):
                                    result_proc = subp.run(cmd, capture_output=True, text=True)

                                    if result_proc.returncode == 0:
                                        st.success("✅ Pre-generation complete!")
                                        st.code(result_proc.stdout)
                                        st.rerun()
                                    else:
                                        st.error("❌ Pre-generation failed")
                                        st.code(result_proc.stderr)
                            else:
                                st.error(f"Labels file not found for run {run_id}")
                        else:
                            st.error("No matching data file found")

                st.markdown("---")

                # Show available clusters using actual IDs from labels
                available_clusters = actual_cluster_ids
                st.info(f"📊 This run has **{n_clusters}** clusters")

                # Automatically use all available clusters
                selected_clusters = available_clusters

                st.markdown("---")

                # Show cached visualizations if they exist
                if cache_exists:
                    st.markdown("### 📊 Cluster Visualizations")

                    st.info(f"Displaying all {len(selected_clusters)} cluster(s)")

                    # Display each cluster in a single column (stacked vertically)
                    for cluster_id in selected_clusters:
                        img_path = cache_dir / f"cluster_{cluster_id}.png"

                        if img_path.exists():
                            cluster_size = cache_metadata['cluster_sizes'].get(cluster_id, '?')
                            st.markdown(f"**Cluster {cluster_id}** (Size: {cluster_size})")
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.warning(f"Image not found for cluster {cluster_id}")


def show_logs():
    """Log viewer page."""
    st.markdown('<p class="main-header">\U0001F4C4 Logs</p>', unsafe_allow_html=True)

    log_files = list(Config.LOGS_DIR.glob("*.log"))

    if not log_files:
        st.warning("\U000026A0 No log files found.")
        return

    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    selected_log = st.selectbox(
        "Select Log File",
        log_files,
        format_func=lambda x: f"{x.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})",
        help="Choose a log file to view. Files are sorted by date (newest first)."
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        tail_lines = st.number_input(
            "Show last N lines",
            10, 10000, 500, 50,
            help="Number of lines to display from the end of the log. Increase for longer runs."
        )

    with col2:
        log_level_filter = st.selectbox(
            "Filter by level",
            ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"],
            help="Show only messages of specific severity. ERROR: failures only. WARNING: potential issues. INFO: general progress. DEBUG: detailed trace."
        )

    with col3:
        search_term = st.text_input(
            "Search in logs",
            "",
            help="Filter log lines containing this text. Case-insensitive. Example: search 'run_id=5' to see specific run logs."
        )

    if selected_log:
        try:
            with open(selected_log, 'r') as f:
                lines = f.readlines()

            # Apply filters
            if log_level_filter != "ALL":
                lines = [l for l in lines if log_level_filter in l]

            if search_term:
                lines = [l for l in lines if search_term.lower() in l.lower()]

            # Get last N lines
            display_lines = lines[-tail_lines:]

            st.info(f"Showing {len(display_lines)} of {len(lines)} lines")

            # Display in code block
            st.code("".join(display_lines), language="log")

            # Download button
            st.download_button(
                label="\U0001F4E5 Download Full Log",
                data="".join(lines),
                file_name=selected_log.name,
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error reading log file: {e}")


def show_model_manager():
    """Model management page."""
    st.markdown('<p class="main-header">\U0001F4BE Model Manager</p>', unsafe_allow_html=True)

    df = get_all_runs()

    if df.empty:
        st.warning("\U000026A0 No models available yet.")
        return

    # Storage statistics
    st.subheader("\U0001F4CA Storage Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        labels_count = len(list(Config.LABELS_DIR.glob("*.npz")))
        st.metric("Label Files", labels_count)

    with col2:
        models_count = len(list(Config.MODELS_DIR.glob("*.pkl")))
        st.metric("Model Files", models_count)

    with col3:
        total_size = sum(f.stat().st_size for f in Config.RESULTS_DIR.rglob("*") if f.is_file())
        st.metric("Total Size", f"{total_size / (1024**2):.1f} MB")

    with col4:
        st.metric("Runs in DB", len(df))

    st.markdown("---")

    # Model browser
    st.subheader("\U0001F50D Model Browser")

    # Add sort and filter options
    col1, col2 = st.columns(2)
    with col1:
        sort_option = st.selectbox(
            "Sort by",
            ['run_id', 'silhouette_score', 'n_clusters'],
            key='model_sort',
            help="Sort models by: Run ID (chronological), Silhouette Score (quality), or Number of Clusters."
        )
    with col2:
        ascending = st.checkbox(
            "Ascending",
            value=False,
            help="Sort order. Unchecked: highest values first. Checked: lowest values first."
        )

    sorted_df = df.sort_values(by=sort_option, ascending=ascending)

    for _, row in sorted_df.iterrows():
        with st.expander(f"Run {row['run_id']} - {row['config_id']} (Silhouette: {row.get('silhouette_score', 'N/A')})"):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**Config:** {row['config_id']}")
                st.write(f"**Clusters:** {row['n_clusters']}, **Noise:** {row['noise_ratio']:.1%}")
                st.write(f"**Timestamp:** {row['timestamp']}")

            with col2:
                # Check if files exist
                label_pattern = f"labels_run{row['run_id']:04d}_*.npz"
                model_pattern = f"clusterer_run{row['run_id']:04d}_*.pkl"

                has_labels = len(list(Config.LABELS_DIR.glob(label_pattern))) > 0
                has_model = len(list(Config.MODELS_DIR.glob(model_pattern))) > 0

                st.write(f"Labels: {'✅' if has_labels else '❌'}")
                st.write(f"Model: {'✅' if has_model else '❌'}")

            with col3:
                if st.button(f"\U0001F5D1 Delete", key=f"del_{row['run_id']}"):
                    if st.session_state.get(f'confirm_del_{row["run_id"]}', False):
                        try:
                            st.session_state.storage.delete_run(row['run_id'])
                            st.success(f"Deleted run {row['run_id']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.session_state[f'confirm_del_{row["run_id"]}'] = True
                        st.warning("Click again to confirm deletion")

    # Cleanup tools
    st.markdown("---")
    st.subheader("\U0001F9F9 Cleanup Tools")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("\U0001F5D1 Delete All Logs", type="secondary"):
            for log_file in Config.LOGS_DIR.glob("*.log"):
                log_file.unlink()
            st.success("All logs deleted")
            st.rerun()

    with col2:
        if st.button("\U000026A0 Delete All Results", type="secondary"):
            if st.session_state.get('confirm_delete_all', False):
                # Delete all result files
                for f in Config.LABELS_DIR.glob("*.npz"):
                    f.unlink()
                for f in Config.MODELS_DIR.glob("*.pkl"):
                    f.unlink()
                for f in Config.METRICS_DIR.glob("*.csv"):
                    f.unlink()
                for f in Config.METRICS_DIR.glob("*.txt"):
                    f.unlink()
                st.success("All results deleted")
                st.rerun()
            else:
                st.session_state['confirm_delete_all'] = True
                st.warning("Click again to confirm deletion of ALL results")


if __name__ == "__main__":
    main()
