import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "C:/Users/hp/OneDrive - Higher Education Commission/Pictures/Giant Python Project/compiled_psx_historical_2017_2025.csv"
RISK_FREE_RATE = 0.08 / 252  # approximate daily risk-free rate (e.g., 8% annual)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)



# ----------------------- Data Loading & Cleaning -----------------------

def load_price_data(
    csv_path,
    normalize: bool = False,
):
    """
    Load historical price data from CSV and perform basic cleaning.

    Expected columns for this project dataset:
    - DATE (or Date)
    - SYMBOL
    - OHLC price columns such as OPEN, HIGH, LOW, CLOSE, VOLUME, etc.

    This function:
    - Reads CSV with robust exception handling.
    - Converts Date to datetime and sets it as index.
    - Sorts by date.
    - Handles missing values via forward-fill, then backward-fill, then drops remaining.
    - Optionally normalizes price columns by rebasing the first valid row to 1.0.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found: {csv_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"[ERROR] CSV file is empty: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Failed to read CSV '{csv_path}': {e}")
        return pd.DataFrame()

    # Handle different possible date column names
    date_col = None
    if "Date" in df.columns:
        date_col = "Date"
    elif "DATE" in df.columns:
        date_col = "DATE"
    else:
        print("[ERROR] CSV must contain a 'Date' or 'DATE' column.")
        return pd.DataFrame()

    # Convert date column to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        print(f"[ERROR] Failed to parse date column '{date_col}': {e}")
        return pd.DataFrame()

    # If SYMBOL and CLOSE columns exist (structure of your PSX CSV),
    # pivot to get one column per asset (symbol) with CLOSE prices.
    if "SYMBOL" in df.columns and "CLOSE" in df.columns:
        price_df = (
            df[[date_col, "SYMBOL", "CLOSE"]]
            .dropna(subset=["CLOSE"])
            .pivot_table(index=date_col, columns="SYMBOL", values="CLOSE")
        )
    else:
        # Fallback: use all numeric columns as generic assets
        df = df.sort_values(date_col).set_index(date_col)
        price_df = df.apply(pd.to_numeric, errors="coerce")

    # Sort by date index
    price_df = price_df.sort_index()

    # Handle missing values with forward-fill then backward-fill
    numeric_df = price_df.ffill().bfill()
    numeric_df = numeric_df.dropna(how="all")

    if numeric_df.empty:
        print("[ERROR] No valid numeric data after cleaning.")
        return pd.DataFrame()

    if normalize:
        # Rebase each column to start at 1.0
        numeric_df = numeric_df / numeric_df.iloc[0]

    return numeric_df


# ---------------------------------------------------------------------------
# Financial Metrics
# ---------------------------------------------------------------------------

def compute_daily_returns(prices: pd.DataFrame):
    """Daily percentage returns from price data."""
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna(how="all")
    return returns


def compute_mean_cov(returns: pd.DataFrame):
    """Mean daily returns and covariance matrix."""
    if returns.empty:
        return np.array([]), np.array([[]])
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    return mean_returns, cov_matrix


def filter_assets(prices, returns, min_days: int = 60):
    """
    Keep only assets with enough data and non-zero volatility.
    This avoids flat / illiquid symbols that break optimization and ML.
    """
    if returns.empty:
        return prices, returns
    vol = returns.std()
    counts = returns.count()
    valid_cols = vol[(vol > 0) & (counts >= min_days)].index
    prices_f = prices[valid_cols]
    returns_f = returns[valid_cols]
    return prices_f, returns_f


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    """Portfolio expected return, volatility, Sharpe ratio."""
    if (
        weights.size == 0
        or mean_returns.size == 0
        or cov_matrix.size == 0
        or len(weights) != len(mean_returns)
    ):
        return np.nan, np.nan, np.nan

    weights = np.array(weights)
    weights = weights / weights.sum()  # ensure weights sum to 1

    port_return = float(np.dot(weights, mean_returns))
    port_variance = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    port_vol = np.sqrt(port_variance) if port_variance >= 0 else np.nan
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan
    return port_return, port_vol, sharpe


# ---------------------------------------------------------------------------
# Portfolio Optimization via Random Simulation
# ---------------------------------------------------------------------------

class PortfolioSimulationResult:
    """Simple container for portfolio simulation results."""

    def __init__(self, returns, volatility, sharpe, weights):
        self.returns = returns
        self.volatility = volatility
        self.sharpe = sharpe
        self.weights = weights  # shape (num_portfolios, num_assets)


def simulate_random_portfolios(
    num_portfolios,
    mean_returns,
    cov_matrix,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PortfolioSimulationResult:
    """
    Simulate random portfolios and compute performance metrics.
    """
    num_assets = len(mean_returns)
    if num_assets == 0 or cov_matrix.size == 0:
        return PortfolioSimulationResult(
            returns=np.array([]),
            volatility=np.array([]),
            sharpe=np.array([]),
            weights=np.array([[]]),
        )

    results_ret = np.zeros(num_portfolios)
    results_vol = np.zeros(num_portfolios)
    results_sharpe = np.zeros(num_portfolios)
    results_weights = np.zeros((num_portfolios, num_assets))

    for i in range(num_portfolios):
        # Random weights that sum to 1
        weights = np.random.rand(num_assets)
        weights /= weights.sum()
        port_ret, port_vol, sharpe = portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        results_weights[i, :] = weights
        results_ret[i] = port_ret
        results_vol[i] = port_vol
        results_sharpe[i] = sharpe

    return PortfolioSimulationResult(
        returns=results_ret,
        volatility=results_vol,
        sharpe=results_sharpe,
        weights=results_weights,
    )


def get_max_sharpe_portfolio(res):
    """Return index of portfolio with maximum Sharpe ratio."""
    if res.sharpe.size == 0 or np.all(np.isnan(res.sharpe)):
        return None
    return int(np.nanargmax(res.sharpe))


def get_min_risk_portfolio(res):
    """Return index of portfolio with minimum volatility."""
    if res.volatility.size == 0 or np.all(np.isnan(res.volatility)):
        return None
    return int(np.nanargmin(res.volatility))


def get_max_return_portfolio(res):
    """Return index of portfolio with maximum return."""
    if res.returns.size == 0 or np.all(np.isnan(res.returns)):
        return None
    return int(np.nanargmax(res.returns))


# ---------------------------------------------------------------------------
# Machine Learning Module
# ---------------------------------------------------------------------------

def prepare_ml_data(series: pd.Series, n_lags: int = 5):
    """Lag features (last n_lags closes) to predict next-day close."""
    values = series.values.astype(float)
    X, y = [], []
    for i in range(n_lags, len(values) - 1):
        X.append(values[i - n_lags : i])
        y.append(values[i + 1])
    return np.array(X), np.array(y)


class MLResult:
    """Container for machine learning model results."""

    def __init__(
        self,
        model_name,
        mse,
        mae,
        r2,
        next_day_pred,
        last_actual,
        trend,
        y_test,
        y_pred,
    ):
        self.model_name = model_name
        self.mse = mse
        self.mae = mae
        self.r2 = r2
        self.next_day_pred = next_day_pred
        self.last_actual = last_actual
        self.trend = trend
        self.y_test = y_test
        self.y_pred = y_pred


def train_and_evaluate_models(
    series: pd.Series,
    use_random_forest: bool = True,
    n_lags: int = 5,
):
    """
    Train Linear Regression (and optionally Random Forest) models
    to predict next-day closing price.
    """
    results = []

    if series is None or series.empty:
        return results

    X, y = prepare_ml_data(series, n_lags=n_lags)
    if X.shape[0] < 20:
        # Not enough data for a meaningful split
        return results

    # Time-series aware split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = [("Linear Regression", LinearRegression())]
    if use_random_forest:
        models.append(("Random Forest", RandomForestRegressor(random_state=RANDOM_SEED)))

    for name, model in models:
        try:
            # Train directly on raw features (no scaling)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = float(mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            # Next-day prediction: use the last available n_lags values
            last_window = series.values[-n_lags:]
            next_day_pred = None
            last_actual = float(series.values[-1])
            trend = None

            if len(last_window) == n_lags:
                next_day_pred = float(model.predict(last_window.reshape(1, -1))[0])
                if next_day_pred > last_actual:
                    trend = "UP"
                elif next_day_pred < last_actual:
                    trend = "DOWN"
                else:
                    trend = "FLAT"

            results.append(
                MLResult(
                    model_name=name,
                    mse=mse,
                    mae=mae,
                    r2=r2,
                    next_day_pred=next_day_pred,
                    last_actual=last_actual,
                    trend=trend,
                    y_test=y_test,
                    y_pred=y_pred,
                )
            )
        except Exception as e:
            print(f"[WARN] Failed to train {name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Visualization Helpers (Matplotlib)
# ---------------------------------------------------------------------------

def plot_price_series(prices: pd.DataFrame, assets):
    """Price vs time for selected assets."""
    fig, ax = plt.subplots(figsize=(10, 4))
    subset = prices[assets]
    subset.plot(ax=ax)
    ax.set_title("Price vs Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_daily_returns(returns: pd.DataFrame, assets):
    """Daily returns over time for selected assets."""
    fig, ax = plt.subplots(figsize=(10, 4))
    subset = returns[assets]
    subset.plot(ax=ax)
    ax.set_title("Daily Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(actual, predicted, title):
    """Predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, predicted, alpha=0.6)
    # Diagonal line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_risk_return_scatter(res, max_sharpe_idx, min_risk_idx, max_return_idx):
    """Risk-return scatter plot for simulated portfolios."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if res.volatility.size > 0 and res.returns.size > 0:
        scatter = ax.scatter(res.volatility, res.returns, c=res.sharpe, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    # Highlight key portfolios
    def highlight(idx, color: str, label: str):
        if idx is not None and 0 <= idx < res.volatility.size:
            ax.scatter(
                res.volatility[idx],
                res.returns[idx],
                color=color,
                marker="*",
                s=200,
                label=label,
                edgecolor="black",
            )

    highlight(max_sharpe_idx, "red", "Max Sharpe")
    highlight(min_risk_idx, "blue", "Min Risk")
    highlight(max_return_idx, "green", "Max Return")

    ax.set_title("Portfolio Risk vs Return")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_sharpe_histogram(res):
    """Histogram of Sharpe ratios."""
    fig, ax = plt.subplots(figsize=(7, 4))
    valid_sharpes = res.sharpe[~np.isnan(res.sharpe)]
    if valid_sharpes.size > 0:
        ax.hist(valid_sharpes, bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of Sharpe Ratios")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_equal_weight_cumulative_returns(returns, assets):
    """Cumulative returns of an equal-weight portfolio of selected assets."""
    if returns.empty or not assets:
        return None
    sub = returns[assets]
    weights = np.ones(len(assets)) / len(assets)
    port_ret = sub.dot(weights)
    cum = (1 + port_ret).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    cum.plot(ax=ax)
    ax.set_title("Equal-Weight Portfolio Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth (× initial)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_actual_predicted_series(actual, predicted, title):
    """Time series of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(actual, label="Actual")
    ax.plot(predicted, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Time (test samples)")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig





# ---------------------------------------------------------------------------
# Console Summary
# ---------------------------------------------------------------------------

def run_console_summary(prices, returns, res, asset_names, ml_results, volatility_dict):
    """Print a concise summary of portfolio performance and ML predictions."""
    print("\n=== DATA SUMMARY ===")
    if prices.empty:
        print("No data loaded.")
        return
    print(f"Rows: {len(prices)}, Columns (assets): {len(prices.columns)}")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")

    missing_counts = prices.isna().sum()
    if missing_counts.sum() > 0:
        print("\nMissing values by column:")
        print(missing_counts)
    else:
        print("\nNo missing values after cleaning.")

    print("\n=== PORTFOLIO OPTIMIZATION ===")
    max_sharpe_idx = get_max_sharpe_portfolio(res)
    min_risk_idx = get_min_risk_portfolio(res)
    max_return_idx = get_max_return_portfolio(res)

    def print_portfolio(label: str, idx):
        if idx is None:
            print(f"{label}: Not available.")
            return
        w = res.weights[idx]
        ret_ = res.returns[idx]
        vol_ = res.volatility[idx]
        sharpe_ = res.sharpe[idx]
        print(f"\n{label}:")
        print(f"  Expected Return: {ret_:.6f}")
        print(f"  Volatility    : {vol_:.6f}")
        print(f"  Sharpe Ratio  : {sharpe_:.3f}")
        print("  Weights:")
        for asset, weight in zip(asset_names, w):
            print(f"    {asset}: {weight:.3f}")

    print_portfolio("Maximum Sharpe Ratio Portfolio", max_sharpe_idx)
    print_portfolio("Minimum Risk Portfolio", min_risk_idx)
    print_portfolio("Highest Return Portfolio", max_return_idx)

    print("\n=== MACHINE LEARNING PREDICTIONS ===")
    if not ml_results:
        print("No ML results available (insufficient data or training failure).")
    else:
        for asset, models in ml_results.items():
            print(f"\nAsset: {asset}")
            for r in models:
                print(f"  Model: {r.model_name}")
                print(f"    MSE : {r.mse:.6f}")
                print(f"    MAE : {r.mae:.6f}")
                print(f"    R^2 : {r.r2:.4f}")
                if r.next_day_pred is not None:
                    print(
                        f"    Next-day prediction: {r.next_day_pred:.4f} "
                        f"(last actual: {r.last_actual:.4f}, trend: {r.trend})"
                    )
                else:
                    print("    Next-day prediction: Not available.")

    print("\n=== ASSETS SORTED BY VOLATILITY ===")
    sorted_vol = sorted(volatility_dict.items(), key=lambda x: x[1])
    for asset, vol in sorted_vol:
        print(f"  {asset}: volatility={vol:.6f}")

    print("\n[INFO] For interactive visualizations and controls, run via Streamlit:")
    print("  streamlit run main.py\n")


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def run_streamlit_app():
    """Build and run the Streamlit GUI."""
    st.set_page_config(page_title="AI Portfolio Optimization & Risk Analysis", layout="wide")
    st.title("AI-Driven Portfolio Optimization & Risk Analysis")

    st.sidebar.header("Configuration")
    normalize = st.sidebar.checkbox("Normalize prices (rebase to 1.0)", value=False)
    num_portfolios = st.sidebar.slider(
        "Number of simulated portfolios", min_value=500, max_value=50000, value=5000, step=500
    )
    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate (annual)", value=0.08, min_value=-0.05, max_value=0.5, step=0.01
    )

    st.sidebar.markdown("---")
    st.sidebar.write("CSV file:")
    st.sidebar.code(CSV_PATH)

    # Load data
    prices = load_price_data(CSV_PATH, normalize=normalize)
    if prices.empty:
        st.error("Failed to load data. Please check that the CSV file is present and valid.")
        return

    # Restrict to last 2 years to keep the app responsive
    max_date_all = prices.index.max()
    two_years_ago = max_date_all - pd.DateOffset(years=2)
    prices = prices[prices.index >= two_years_ago]
    if prices.empty:
        st.error("No data available in the last 2 years.")
        return

    # Limit date range (optional)
    min_date, max_date = prices.index.min(), prices.index.max()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date()
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask = (prices.index >= start_date) & (prices.index <= end_date)
        prices = prices.loc[mask]

    if prices.empty:
        st.error("No data in selected date range.")
        return

    asset_names = list(prices.columns)
    returns = compute_daily_returns(prices)
    # Filter out illiquid / flat assets to make optimization meaningful
    prices, returns = filter_assets(prices, returns, min_days=60)
    asset_names = list(prices.columns)
    if len(asset_names) == 0:
        st.error("No assets with sufficient data/volatility in the selected period.")
        return
    mean_returns, cov_matrix = compute_mean_cov(returns)

    # Dictionaries for latest price and volatility
    latest_price_dict = {a: float(prices[a].iloc[-1]) for a in asset_names}
    volatility_dict = {
        a: float(returns[a].std()) if a in returns.columns else np.nan for a in asset_names
    }

    # Simulate portfolios
    sim_res = simulate_random_portfolios(
        num_portfolios=num_portfolios,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate / 252.0,
    )

    # ML: user selects one asset
    st.sidebar.markdown("---")
    st.sidebar.header("Machine Learning")
    asset_for_ml = st.sidebar.selectbox("Select asset for ML prediction", asset_names)

    ml_results_for_asset = {}
    if asset_for_ml in prices.columns:
        series = prices[asset_for_ml]
        ml_models = train_and_evaluate_models(series, use_random_forest=True, n_lags=5)
        if ml_models:
            ml_results_for_asset[asset_for_ml] = ml_models

    # ------------------ Main Layout ------------------
    tab_overview, tab_prices, tab_portfolio, tab_ml, tab_structures = st.tabs(
        ["Data Overview", "Prices & Returns", "Portfolio Optimization", "Machine Learning", "Data Structures"]
    )

    with tab_overview:
        st.subheader("Dataset Overview")
        st.write(f"Rows: {len(prices)}, Columns (assets): {len(prices.columns)}")
        st.write(f"Date range: {prices.index.min().date()} → {prices.index.max().date()}")
        st.dataframe(prices.head())
        st.markdown("**Basic Statistics (last available prices)**")
        st.dataframe(prices.describe())

        st.markdown("**Missing Value Summary (after cleaning)**")
        st.dataframe(prices.isna().sum())

    with tab_prices:
        st.subheader("Price and Return Visualizations")
        selected_assets = st.multiselect("Select assets to display", asset_names, default=asset_names[: min(3, len(asset_names))])
        if not selected_assets:
            st.warning("Please select at least one asset.")
        else:
            st.pyplot(plot_price_series(prices, selected_assets))
            if not returns.empty:
                st.pyplot(plot_daily_returns(returns, selected_assets))
                eq_fig = plot_equal_weight_cumulative_returns(returns, selected_assets)
                if eq_fig is not None:
                    st.pyplot(eq_fig)
            else:
                st.warning("Not enough data to compute returns.")

    with tab_portfolio:
        st.subheader("Portfolio Optimization Results")
        max_sharpe_idx = get_max_sharpe_portfolio(sim_res)
        min_risk_idx = get_min_risk_portfolio(sim_res)
        max_return_idx = get_max_return_portfolio(sim_res)

        st.pyplot(plot_risk_return_scatter(sim_res, max_sharpe_idx, min_risk_idx, max_return_idx))
        st.pyplot(plot_sharpe_histogram(sim_res))

        def show_portfolio(label, idx):
            st.markdown(f"### {label}")
            if idx is None:
                st.info("Not available.")
                return
            data = {
                "Asset": asset_names,
                "Weight": sim_res.weights[idx],
            }
            port_df = pd.DataFrame(data)
            st.dataframe(port_df.style.format({"Weight": "{:.3f}"}))
            st.write(
                f"**Expected Return:** {sim_res.returns[idx]:.6f}  |  "
                f"**Volatility:** {sim_res.volatility[idx]:.6f}  |  "
                f"**Sharpe Ratio:** {sim_res.sharpe[idx]:.3f}"
            )

        show_portfolio("Maximum Sharpe Ratio Portfolio", max_sharpe_idx)
        show_portfolio("Minimum Risk Portfolio", min_risk_idx)
        show_portfolio("Highest Return Portfolio", max_return_idx)

    with tab_ml:
        st.subheader("Next-Day Price Prediction (Supervised ML)")
        if not ml_results_for_asset:
            st.warning(
                "ML results not available. You may need a longer time span or more data points for the selected asset."
            )
        else:
            for asset, models in ml_results_for_asset.items():
                st.markdown(f"### Asset: {asset}")
                for r in models:
                    st.markdown(f"**Model:** {r.model_name}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{r.mse:.6f}")
                    col2.metric("MAE", f"{r.mae:.6f}")
                    col3.metric("R²", f"{r.r2:.4f}")

                    if r.next_day_pred is not None:
                        st.write(
                            f"**Next-day predicted close:** {r.next_day_pred:.4f} "
                            f"(last actual: {r.last_actual:.4f}, trend: **{r.trend}**)"
                        )
                    else:
                        st.write("Next-day prediction not available.")

                    if r.y_test.size > 0:
                        fig = plot_predicted_vs_actual(r.y_test, r.y_pred, f"{asset} - {r.model_name}")
                        st.pyplot(fig)
                        ts_fig = plot_actual_predicted_series(r.y_test, r.y_pred, f"{asset} - {r.model_name} (Time series)")
                        st.pyplot(ts_fig)

    with tab_structures:
        st.subheader("Data Structures for Analysis")

        st.markdown("### Asset → Latest Price (Dictionary)")
        price_df = pd.DataFrame(
            {"Asset": list(latest_price_dict.keys()), "Latest Price": list(latest_price_dict.values())}
        )
        st.dataframe(price_df)

        st.markdown("### Assets Sorted by Volatility")
        vol_pairs = sorted(volatility_dict.items(), key=lambda x: x[1])
        vol_df = pd.DataFrame(vol_pairs, columns=["Asset", "Volatility"])
        st.dataframe(vol_df.style.format({"Volatility": "{:.6f}"}))


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Run console summary when executed directly."""
    print("[INFO] Loading data...")
    prices = load_price_data(CSV_PATH, normalize=False)
    if prices.empty:
        print("[FATAL] Could not load data. Exiting.")
        return

    # Use only last 2 years of data for faster computations
    max_date_all = prices.index.max()
    two_years_ago = max_date_all - pd.DateOffset(years=2)
    prices = prices[prices.index >= two_years_ago]
    if prices.empty:
        print("[FATAL] No data available in the last 2 years. Exiting.")
        return

    returns = compute_daily_returns(prices)
    mean_returns, cov_matrix = compute_mean_cov(returns)
    # Filter out illiquid / flat assets
    prices, returns = filter_assets(prices, returns, min_days=60)
    asset_names = list(prices.columns)
    if len(asset_names) == 0:
        print("[FATAL] No assets with sufficient data/volatility in last 2 years. Exiting.")
        return

    # Build dictionaries and volatility summary
    latest_price_dict = {a: float(prices[a].iloc[-1]) for a in asset_names}
    volatility_dict = {
        a: float(returns[a].std()) if a in returns.columns else np.nan for a in asset_names
    }

    print("[INFO] Simulating random portfolios...")
    sim_res = simulate_random_portfolios(
        num_portfolios=3000,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=RISK_FREE_RATE,
    )

    # ML for a default asset: first column
    ml_results = {}
    if asset_names:
        default_asset = asset_names[0]
        print(f"[INFO] Training ML models for asset: {default_asset}")
        series = prices[default_asset]
        models = train_and_evaluate_models(series, use_random_forest=True, n_lags=5)
        if models:
            ml_results[default_asset] = models

    run_console_summary(
        prices=prices,
        returns=returns,
        res=sim_res,
        asset_names=asset_names,
        ml_results=ml_results,
        volatility_dict=volatility_dict,
    )


if __name__ == "__main__":
    # Run Streamlit app automatically when the script is executed
    run_streamlit_app()


