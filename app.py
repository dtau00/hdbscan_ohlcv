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

            # File selector
            selected_file_names = st.multiselect(
                "Select CSV files to process",
                file_names,
                default=st.session_state.selected_files,
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

        if data_files and selected_file_names:
            total_runs = total_configs * len(selected_file_names)
            st.info(f"📊 This will run **{total_configs}** configurations × **{len(selected_file_names)}** file(s) = **{total_runs}** total runs")
        else:
            st.info(f"ℹ️ This will run **{total_configs}** configurations")

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

        # Yahoo Finance Download Section
        st.markdown("### \U0001F4E5 Download from Yahoo Finance")

        # Tabs for single vs bulk download
        download_tabs = st.tabs(["\U0001F4C4 Single Ticker", "\U0001F4CA Bulk Download"])

        # Tab 1: Single Ticker Download
        with download_tabs[0]:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                yf_ticker = st.text_input(
                    "Ticker Symbol",
                    "SPY",
                    help="Stock ticker symbol (e.g., SPY, AAPL, QQQ). Works with stocks, ETFs, and some futures."
                )

            with col2:
                yf_interval = st.selectbox(
                    "Interval",
                    ["1m", "5m", "15m", "30m", "1h", "1d"],
                    index=0,
                    help="Bar timeframe. 1m: 1-minute bars (max 7 days). Higher intervals allow longer history."
                )

            with col3:
                # Dynamic period options based on interval
                interval_limits = {
                    "1m": ["1d", "5d", "7d", "1mo"],  # 1mo sometimes works, 7d is reliable
                    "5m": ["1d", "5d", "1mo", "3mo"],
                    "15m": ["1d", "5d", "1mo", "3mo", "6mo"],
                    "30m": ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                    "1h": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
                }

                yf_period = st.selectbox(
                    "Period",
                    interval_limits.get(yf_interval, ["7d"]),
                    help=f"Historical period to download. Max for {yf_interval}: {interval_limits.get(yf_interval, ['7d'])[-1]}"
                )

            with col4:
                yf_run_immediately = st.checkbox(
                    "Run Clustering After Download",
                    value=True,
                    help="Automatically run HDBSCAN clustering on downloaded data using default parameters."
                )

            col1, col2 = st.columns([3, 1])

            with col1:
                if st.button("\U0001F4E5 Download & Save Data", type="primary", width='stretch'):
                    download_from_yahoo(yf_ticker, yf_interval, yf_period, yf_run_immediately)

            with col2:
                st.info(f"ℹ️ Est. bars: {estimate_bars(yf_interval, yf_period)}")

        # Tab 2: Bulk Download (NASDAQ-100)
        with download_tabs[1]:
            st.markdown("Download data for all NASDAQ-100 stocks")

            # NASDAQ-100 ticker list (current as of 2025)
            nas100_tickers = [
                "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
                "ASML", "AMD", "PEP", "ADBE", "CSCO", "CMCSA", "TMUS", "NFLX", "INTC", "QCOM",
                "INTU", "TXN", "AMGN", "HON", "AMAT", "BKNG", "SBUX", "GILD", "ADI", "VRTX",
                "PYPL", "ADP", "ISRG", "LRCX", "REGN", "MU", "MDLZ", "PANW", "KLAC", "MELI",
                "SNPS", "CDNS", "MAR", "CSX", "ORLY", "ABNB", "MNST", "FTNT", "ADSK", "MRVL",
                "CHTR", "NXPI", "AEP", "WDAY", "DASH", "PCAR", "KDP", "PAYX", "CPRT", "DXCM",
                "MRNA", "ROST", "ODFL", "EA", "CTSH", "FAST", "CEG", "LULU", "IDXX", "KHC",
                "EXC", "GEHC", "TTD", "TEAM", "XEL", "VRSK", "CTAS", "FANG", "BKR", "ANSS",
                "ZS", "DDOG", "ON", "BIIB", "CCEP", "ILMN", "CDW", "GFS", "WBD", "MDB",
                "SMCI", "CRWD", "APP", "TTWO", "WBA", "DLTR", "PDD", "ZM", "MCHP", "ENPH"
            ]

            col1, col2, col3 = st.columns(3)

            with col1:
                bulk_interval = st.selectbox(
                    "Interval",
                    ["1m", "5m", "15m", "30m", "1h", "1d"],
                    index=5,  # Default to 1d for bulk
                    key="bulk_interval",
                    help="Bar timeframe. For bulk downloads, 1d is recommended to avoid API limits."
                )

            with col2:
                interval_limits_bulk = {
                    "1m": ["1d", "5d", "7d"],
                    "5m": ["1d", "5d", "1mo"],
                    "15m": ["1d", "5d", "1mo", "3mo"],
                    "30m": ["1d", "5d", "1mo", "3mo"],
                    "1h": ["1d", "5d", "1mo", "3mo", "6mo"],
                    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
                }

                bulk_period = st.selectbox(
                    "Period",
                    interval_limits_bulk.get(bulk_interval, ["1mo"]),
                    index=3 if bulk_interval == "1d" else 0,  # Default to 1y for 1d data
                    key="bulk_period",
                    help="Historical period. Shorter periods recommended for bulk downloads."
                )

            with col3:
                max_tickers = st.number_input(
                    "Max Tickers",
                    1, len(nas100_tickers), len(nas100_tickers),
                    key="max_tickers",
                    help="Maximum number of tickers to download (in alphabetical order). Set to 100 for all."
                )

            # Stock selection
            st.markdown("**Select Stocks:**")
            selection_mode = st.radio(
                "Mode",
                ["All", "Custom"],
                horizontal=True,
                help="All: download all NAS100 stocks. Custom: select specific tickers."
            )

            if selection_mode == "Custom":
                selected_tickers = st.multiselect(
                    "Select tickers",
                    nas100_tickers,
                    default=nas100_tickers[:10],
                    help="Choose which tickers to download."
                )
            else:
                selected_tickers = nas100_tickers[:max_tickers]

            st.info(f"📊 Will download {len(selected_tickers)} stocks × ~{estimate_bars(bulk_interval, bulk_period)} bars each")

            # Options
            col1, col2 = st.columns(2)
            with col1:
                skip_errors = st.checkbox(
                    "Skip Errors",
                    value=True,
                    help="Continue downloading if some tickers fail. Recommended for bulk downloads."
                )
            with col2:
                combine_data = st.checkbox(
                    "Combine into Single File",
                    value=False,
                    help="Save all data to a single CSV with a 'Ticker' column. Otherwise, separate files per ticker."
                )

            # Download button
            if st.button("📥 Start Bulk Download", type="primary", width='stretch'):
                download_bulk_nas100(selected_tickers, bulk_interval, bulk_period, skip_errors, combine_data)

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


def estimate_bars(interval, period):
    """Estimate number of bars for given interval and period."""
    bars_per_day = {
        "1m": 390,    # US market hours
        "5m": 78,
        "15m": 26,
        "30m": 13,
        "1h": 6.5,
        "1d": 1
    }

    days = {
        "1d": 1,
        "5d": 5,
        "7d": 7,
        "1mo": 20,
        "3mo": 60,
        "6mo": 120,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
        "10y": 2520,
        "max": 25200
    }

    bars_day = bars_per_day.get(interval, 390)
    num_days = days.get(period, 7)

    estimated = int(bars_day * num_days)
    return f"~{estimated:,}"


def download_from_yahoo(ticker: str, interval: str, period: str, run_clustering: bool = True) -> None:
    """Download data from Yahoo Finance and optionally run clustering."""
    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance not installed. Run: pip install yfinance")
        return

    # Expected minimum bars for 1m interval with 1mo period
    min_bars_for_1mo = 5000  # ~13 trading days

    with st.spinner(f"Downloading {ticker} {interval} data for {period}..."):
        try:
            # Download data
            data: pd.DataFrame | None = yf.download(ticker, interval=interval, period=period, progress=False)

            # Check if data is None or empty
            if data is None or data.empty:
                st.error(f"No data returned for {ticker}. Check ticker symbol and try again.")
                return

            # At this point, data is guaranteed to be a non-empty DataFrame
            # Fallback logic for 1m interval with 1mo period
            if interval == "1m" and period == "1mo" and len(data) < min_bars_for_1mo:
                st.warning(f"⚠️ 1mo request returned only {len(data)} bars (expected ~7,800). Retrying with 7d fallback...")
                fallback_data: pd.DataFrame | None = yf.download(ticker, interval=interval, period="7d", progress=False)
                period = "7d (fallback)"  # Update for filename

                # Check fallback data
                if fallback_data is None or fallback_data.empty:
                    st.error(f"No data returned for {ticker}. Check ticker symbol and try again.")
                    return

                data = fallback_data
                st.info(f"ℹ️ Using 7d fallback: {len(data)} bars retrieved")

            # Flatten multi-level columns if present (happens with single ticker downloads)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure we have OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                st.error(f"Downloaded data missing required columns. Got: {list(data.columns)}")
                return

            # Keep only OHLCV and ensure proper order
            data = data[required_cols].copy()

            # Remove any NaN rows
            data = data.dropna()

            # Show success message with actual vs expected
            if interval == "1m" and "1mo" in str(period):
                expected_bars = 7800
                if len(data) >= min_bars_for_1mo:
                    st.success(f"✅ Got full month of data: {len(data):,} bars (expected ~{expected_bars:,})")
                else:
                    st.info(f"ℹ️ Limited data: {len(data):,} bars (~7 days worth)")

            # Save to file
            filename = f"{ticker}_{interval}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_path = Config.DATA_DIR / filename
            data.to_csv(save_path)

            st.success(f"✅ Downloaded {len(data):,} bars to {filename}")

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
                st.metric("Total Bars", f"{len(data):,}")
            with col2:
                st.metric("Date Range", f"{len(data.index)} bars")
            with col3:
                st.metric("Start", data.index[0].strftime('%Y-%m-%d'))
            with col4:
                st.metric("End", data.index[-1].strftime('%Y-%m-%d'))

            # Run clustering if requested
            if run_clustering:
                st.markdown("---")
                st.markdown("### \U000025B6 Running Clustering on Downloaded Data")

                # Use default parameters for quick clustering
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


def download_bulk_nas100(tickers, interval, period, skip_errors=True, combine_data=False):
    """Download data for multiple NASDAQ-100 tickers in bulk."""
    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance not installed. Run: pip install yfinance")
        return

    total_tickers = len(tickers)
    successful_downloads = []
    failed_downloads = []
    all_data = []

    st.info(f"Starting bulk download of {total_tickers} tickers...")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    for i, ticker in enumerate(tickers):
        status_text.text(f"Downloading {i+1}/{total_tickers}: {ticker}")

        try:
            # Download data
            data = yf.download(ticker, interval=interval, period=period, progress=False)

            if data is None or data.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns for {ticker}")

            # Keep only OHLCV
            data = data[required_cols].copy()
            data = data.dropna()

            if len(data) == 0:
                raise ValueError(f"No valid data after cleaning for {ticker}")

            # Save or accumulate
            if combine_data:
                # Add ticker column and accumulate
                data['Ticker'] = ticker
                all_data.append(data)
            else:
                # Save individual file
                filename = f"{ticker}_{interval}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                save_path = Config.DATA_DIR / filename
                data.to_csv(save_path)

            successful_downloads.append((ticker, len(data)))

        except Exception as e:
            failed_downloads.append((ticker, str(e)))
            if not skip_errors:
                st.error(f"Failed on {ticker}: {e}")
                break

        # Update progress
        progress_bar.progress((i + 1) / total_tickers)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Save combined data if requested
    if combine_data and all_data:
        combined_df = pd.concat(all_data, axis=0)
        filename = f"NAS100_bulk_{interval}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = Config.DATA_DIR / filename
        combined_df.to_csv(save_path)
        st.success(f"✅ Saved combined data to {filename} ({len(combined_df):,} total bars)")

    # Show summary
    with results_container:
        st.markdown("### Download Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Successful", len(successful_downloads), delta=None)
        with col2:
            st.metric("Failed", len(failed_downloads), delta=None)
        with col3:
            success_rate = (len(successful_downloads) / total_tickers * 100) if total_tickers > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # Successful downloads details
        if successful_downloads:
            st.markdown("**✅ Successful Downloads:**")
            success_df = pd.DataFrame(successful_downloads, columns=['Ticker', 'Bars'])
            success_df['Total Bars'] = success_df['Bars']
            st.dataframe(success_df, width='stretch', hide_index=True)

            total_bars = success_df['Bars'].sum()
            avg_bars = success_df['Bars'].mean()
            st.info(f"📊 Total: {total_bars:,} bars | Average: {avg_bars:.0f} bars per ticker")

        # Failed downloads details
        if failed_downloads:
            st.markdown("**❌ Failed Downloads:**")
            with st.expander(f"Show {len(failed_downloads)} failed tickers"):
                failed_df = pd.DataFrame(failed_downloads, columns=['Ticker', 'Error'])
                st.dataframe(failed_df, width='stretch', hide_index=True)

        # Next steps
        if successful_downloads and not combine_data:
            st.success(f"✅ Downloaded {len(successful_downloads)} files to `{Config.DATA_DIR}`")
            st.info("💡 Tip: Use the Grid Search tab to cluster individual ticker files.")


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

        with st.spinner(f"Running parallel grid search with {n_jobs} workers..."):
            all_results = parallel_multi_file_grid_search(
                configs=configs,
                data_files=data_file_tuples,
                backend_type=st.session_state.backend_type,
                backend_module=st.session_state.backend_module,
                storage=st.session_state.storage,
                process_func=process_single_config,
                n_jobs=n_jobs,
                verbose=10  # Show progress
            )
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
                # Show available clusters using actual IDs from labels
                available_clusters = actual_cluster_ids
                st.info(f"📊 This run has **{n_clusters}** clusters (IDs: {', '.join(map(str, available_clusters))})")

                # Initialize or reset session state for selected clusters
                # Reset if run_id changed or if stored clusters are invalid for this run
                if ('selected_clusters' not in st.session_state or
                    'last_viz_run_id' not in st.session_state or
                    st.session_state.last_viz_run_id != run_id or
                    not all(c in available_clusters for c in st.session_state.selected_clusters)):
                    st.session_state.selected_clusters = available_clusters[:min(3, n_clusters)]
                    st.session_state.last_viz_run_id = run_id

                # Use multiselect for cluster selection
                selected_clusters = st.multiselect(
                    "Select Cluster IDs",
                    options=available_clusters,
                    default=st.session_state.selected_clusters,
                    help="Select which clusters to visualize. You can choose multiple clusters to compare their patterns."
                )

                # Update session state
                st.session_state.selected_clusters = selected_clusters

                # Select/Deselect buttons in a row
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Select All", key="select_all_clusters", width='stretch'):
                        st.session_state.selected_clusters = available_clusters.copy()
                        st.rerun()

                with col2:
                    if st.button("❌ Deselect All", key="deselect_all_clusters", width='stretch'):
                        st.session_state.selected_clusters = []
                        st.rerun()

                # Samples slider
                n_samples = st.slider(
                    "Samples per Cluster",
                    1, 50, 5,
                    help="How many random examples to show from each cluster. More samples give better sense of cluster variation."
                )

                # Only show button if clusters are selected
                if not selected_clusters:
                    st.warning("⚠️ Please select at least one cluster to visualize.")
                elif st.button(
                    "Generate Pattern Visualization",
                    help="Creates candlestick charts showing actual OHLCV patterns from each cluster."
                ):
                    try:
                        from tools.visualize_clusters import ClusterVisualizer

                        cluster_list = selected_clusters

                        with st.spinner("Generating patterns..."):
                            visualizer = ClusterVisualizer(results_dir=str(Config.RESULTS_DIR))

                            # Try to determine correct data by checking expected window count
                            # Load labels to get expected count
                            from src.storage import ResultsStorage
                            temp_storage = ResultsStorage()
                            labels, config = temp_storage.load_labels(int(run_id))
                            expected_windows = len(labels)
                            window_size = config['window_size']
                            expected_bars = expected_windows + window_size - 1

                            st.info(f"Looking for data file with ~{expected_bars} bars (creates {expected_windows} windows with window_size={window_size})")

                            # Check data files
                            data_files = list(Config.DATA_DIR.glob("*.csv"))
                            best_match = None
                            min_diff = float('inf')

                            if data_files:
                                for data_file in data_files:
                                    try:
                                        df_temp = pd.read_csv(data_file)
                                        n_bars = len(df_temp)
                                        diff = abs(n_bars - expected_bars)
                                        if diff < min_diff:
                                            min_diff = diff
                                            best_match = data_file
                                    except:
                                        continue

                                if best_match:
                                    ohlcv_df = pd.read_csv(best_match)
                                    st.info(f"Using data from: {best_match.name} ({len(ohlcv_df)} bars)")
                                else:
                                    st.warning("No suitable data file found, using first available")
                                    ohlcv_df = pd.read_csv(data_files[0])
                                    st.info(f"Using data from: {data_files[0].name} ({len(ohlcv_df)} bars)")
                            else:
                                # Generate synthetic data
                                from main import generate_synthetic_ohlcv
                                ohlcv_df = generate_synthetic_ohlcv(n_bars=expected_bars, seed=42)
                                st.info(f"Using synthetic OHLCV data ({expected_bars} bars)")

                            # Generate output path
                            output_path = Config.RESULTS_DIR / "visualizations" / f"clusters_run{int(run_id):04d}.png"

                            # Plot cluster samples
                            result_path = visualizer.plot_cluster_samples(
                                run_id=int(run_id),
                                ohlcv_df=ohlcv_df,
                                cluster_ids=cluster_list,
                                n_samples=n_samples,
                                output_path=str(output_path),
                                show=False
                            )

                            st.success(f"Patterns saved to {result_path}")
                            st.image(str(result_path))
                    except Exception as e:
                        st.error(f"Error generating patterns: {e}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())


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
