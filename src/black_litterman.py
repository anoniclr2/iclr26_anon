import yfinance as yf
import pandas as pd
import numpy as np


from pypfopt import black_litterman, risk_models
from pypfopt import EfficientFrontier, objective_functions


def get_covariance_matrix(price_df: pd.DataFrame, ref_date, window_days: int = 252) -> pd.DataFrame:
    """
    Compute the covariance matrix of asset prices over a rolling window ending at ref_date.

    Parameters:
        price_df (pd.DataFrame): DataFrame of asset prices indexed by datetime.
        ref_date (str or pd.Timestamp): Reference date for which to compute the covariance.
        window_days (int): Number of days (window size) to use for estimation, defaults to 252.

    Returns:
        pd.DataFrame: Covariance matrix computed using Ledoit-Wolf shrinkage.
    
    Raises:
        ValueError: If the ref_date is not in the index.
    """
    # Ensure ref_date is a Timestamp
    if isinstance(ref_date, str):
        ref_date = pd.to_datetime(ref_date)
    
    start_date = (ref_date - pd.DateOffset(days=window_days)).strftime('%Y-%m-%d')
    end_date = ref_date.strftime('%Y-%m-%d')

    if start_date not in price_df.index and end_date not in price_df.index:
        raise ValueError(f"Reference date {ref_date} not found in DataFrame index.")
    
    window_df = price_df.loc[start_date:end_date]
    
    sigma = risk_models.CovarianceShrinkage(window_df).ledoit_wolf()
    return sigma


def get_market_caps(price_df: pd.DataFrame, universe_tickers: list) -> pd.DataFrame:
    """
    Calculate daily market capitalizations.

    Parameters:
        price_df (pd.DataFrame): DataFrame of asset prices where columns are tickers.
        universe_tickers (list): List of tickers for which to fetch shares outstanding.

    Returns:
        pd.DataFrame: DataFrame of market caps with the same shape as price_df.
    """
    outstanding_shares = {}
    for ticker in universe_tickers:
        try:
            stock = yf.Ticker(ticker)
            shares_outstanding = stock.info['sharesOutstanding']
            outstanding_shares[ticker] = shares_outstanding
        except KeyError:
            print(f"{ticker}: Shares outstanding data not available")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    # Build a DataFrame from the outstanding shares; ensure order matches the price_df columns.
    df_outstanding = pd.DataFrame.from_dict(outstanding_shares, orient='index', 
                                            columns=['Shares Outstanding'])
    # Align to the universe tickers as represented in the price_df columns
    df_outstanding = df_outstanding.loc[price_df.columns]
    
    # Multiply prices with shares to get market cap
    df_market_cap = price_df.multiply(df_outstanding['Shares Outstanding'], axis='columns')
    return df_market_cap


def download_market_data(start_date: str, end_date: str, ticker: str = "SPY") -> pd.Series:
    """
    Download market portfolio data (e.g. SPY) for a given date range.
    The start_date is shifted one year back to have sufficient history.

    Parameters:
        start_date (str): Start date (will be shifted 1 year back for calculation).
        end_date (str): End date.
        ticker (str): Ticker symbol for the market portfolio, defaults to "SPY".

    Returns:
        pd.Series: Series of closing prices indexed by datetime.
    """
    # Shift start_date one year back for historical data
    start_dt = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    market_data_df = yf.download(ticker, start=start_str, end=end_str)['Close']
    # Ensure a Series is returned
    return market_data_df.squeeze()


def filter_market_data(market_prices: pd.Series, ref_date, ticker: str = "SPY") -> pd.Series:
    """
    Filter the market portfolio prices to the one-year window ending at ref_date.
    This Series is used to compute the market-implied risk aversion.

    Parameters:
        market_prices (pd.Series): Series of market prices indexed by datetime.
        ref_date (str or pd.Timestamp): Reference date.
        ticker (str): Ticker used in naming (for clarity), defaults to "SPY".

    Returns:
        pd.Series: Filtered market prices for the past year.
    """
    ref_date_ts = pd.to_datetime(ref_date)
    start_date = (ref_date_ts - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    end_date = ref_date_ts.strftime('%Y-%m-%d')
    filtered_prices = market_prices.loc[start_date:end_date]
    # Optionally, rename the Series for clarity (not strictly required)
    filtered_prices.name = f"{ticker}_Close"
    return filtered_prices


def uq_to_confidence(uncertainty: np.ndarray, tickers: list, start_date: str, end_date: str, boost: float = 0.0) -> pd.DataFrame:
    """
    Convert uncertainty measures to confidence levels for each asset over a date range.
    The conversion is done by normalizing uncertainty and subtracting from one.

    Parameters:
        uncertainty (np.ndarray): 2D array of uncertainties with shape (n_days, len(tickers)).
        tickers (list): List of tickers corresponding to the columns.
        start_date (str): Start date for the index.
        end_date (str): End date for the index.

    Returns:
        pd.DataFrame: DataFrame of view confidences with index of dates and columns tickers.
    """
    max_value = np.max(uncertainty)
    normalized_uncertainty = uncertainty / max_value
    confidence = 1 - normalized_uncertainty
    # Boost the confidence by min(confidence + boost, 1)
    confidence = np.clip(confidence + boost, 0, 1)
    # It is assumed that uncertainty has as many rows as there are business days in the range.
    date_index = pd.date_range(start=start_date, end=end_date, freq='B')
    confidence_df = pd.DataFrame(confidence, columns=tickers, index=date_index)
    return confidence_df


def thresholds_to_views(preds: np.ndarray, threshold: float, tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Convert discrete predictions to absolute views based on a threshold.
    For each prediction:
      - 1 is mapped to +threshold,
      - -1 is mapped to -threshold,
      - 0 (or other values) are mapped to 0.

    Parameters:
        preds (np.ndarray): 2D array of predictions with shape (n_days, len(tickers)).
        threshold (float): Threshold value for positive/negative views.
        tickers (list): List of tickers corresponding to the predictions.
        start_date (str): Start date for the index.
        end_date (str): End date for the index.

    Returns:
        pd.DataFrame: DataFrame of absolute views with index as dates and columns as tickers.
    """
    date_index = pd.date_range(start=start_date, end=end_date, freq='B')
    views_df = pd.DataFrame(preds, columns=tickers, index=date_index)
    views_df = views_df.apply(lambda x: np.where(x == 1, threshold,
                                                  np.where(x == -1, -threshold, 0)))
    return views_df


def daily_black_litterman(price_df: pd.DataFrame, ref_date, views: pd.DataFrame, confidence: pd.DataFrame, 
                          market_caps: pd.DataFrame, tau: float, market_data: pd.Series,
                          use_prev_prior_mean: bool = False, use_prev_prior_cov: bool = False, 
                          prev_posterior: pd.Series = None, prev_cov: pd.DataFrame = None, 
                          use_non_convex: bool = False, l2: float = 0.1) -> dict:
    """
    Run the daily Black–Litterman model for a given date.

    Parameters:
        price_df (pd.DataFrame): Price data for the full universe (indexed by date).
        ref_date (str or pd.Timestamp): The date for which to run the model.
        views (pd.DataFrame): DataFrame of absolute views (rows: dates, columns: view tickers).
        confidence (pd.DataFrame): DataFrame of view confidences (same shape as views).
        market_caps (pd.DataFrame): DataFrame of market capitalizations (indexed by date, columns: universe tickers).
        tau (float): Scalar weight on the prior.
        market_data (pd.Series): Market portfolio prices (e.g. SPY) as a Series.
        use_prev_prior_mean (bool): Whether to use the previous posterior returns as prior.
        use_prev_prior_cov (bool): Whether to use the previous covariance as the prior covariance.
        prev_posterior (pd.Series, optional): Previous posterior returns.
        prev_cov (pd.DataFrame, optional): Previous posterior covariance matrix.
        use_non_convex (bool): Flag to use the nonconvex (scipy-based) optimizer for maximizing Sharpe.
                                Default is False (uses the built-in convex optimizer).
        l2 (float): Coefficient for L2 regularization. Set to 0 to turn off regularization.
    
    Returns:
        dict: Dictionary containing:
            - "Posterior mean": pd.Series of posterior expected returns.
            - "Posterior covariance": pd.DataFrame of posterior covariance.
            - "Optimal Portfolio Weights BL": dict of BL optimal weights.
            - "Optimal Portfolio Weights EF": dict of efficient frontier optimal weights.
    """
    # ------ HELPER function for regularization -----
    def sharpe_with_l2(w, expected_returns, cov_matrix, l2_coef=0.1):
        sharpe = objective_functions.sharpe_ratio(w, expected_returns, cov_matrix)
        penalty = l2_coef * np.dot(w, w)  # L2 penalty term
        return sharpe + penalty
    # --------------------------------------------

    # Get the covariance matrix (or reuse previous if specified)
    if use_prev_prior_cov and prev_cov is not None:
        sigma = prev_cov
    else:
        sigma = get_covariance_matrix(price_df, ref_date)

    market_cap_on_ref = market_caps.loc[ref_date]
    market_prices = filter_market_data(market_data, ref_date)
    if isinstance(market_prices, pd.DataFrame):
        market_prices = market_prices.squeeze()

    # "SPY" over the last year: Compute market-implied risk aversion parameter
    delta = black_litterman.market_implied_risk_aversion(market_prices)

    # Compute the market-implied prior returns.
    market_pi = black_litterman.market_implied_prior_returns(market_cap_on_ref, delta, sigma)
    if use_prev_prior_mean and prev_posterior is not None:
        pi = 0.5 * prev_posterior + 0.5 * market_pi
    else:
        pi = market_pi

    # Case 1: When there are no views provided, optimize using the Efficient Frontier on the prior.
    if views is None or confidence is None:
        ef = EfficientFrontier(pi, sigma)
        if use_non_convex:
            weights = ef.nonconvex_objective(
                sharpe_with_l2,
                objective_args=(ef.expected_returns, ef.cov_matrix, l2),
                weights_sum_to_one=True,
            )
            if isinstance(weights, dict):
                optimal_weights = weights
            else:
                optimal_weights = ef.clean_weights(weights)
        else:
            if l2 > 0:
                ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            optimal_weights = ef.clean_weights()
        return {"Posterior mean": pi,
                "Posterior covariance": sigma,
                "Optimal Portfolio Weights EF": optimal_weights}

    # Case 2: When views and confidences are provided, build the Black–Litterman model.
    views_on_ref = views.loc[ref_date]
    confidence_on_ref = confidence.loc[ref_date]
    bl_model = black_litterman.BlackLittermanModel(
        cov_matrix=sigma,
        pi=pi,
        absolute_views=views_on_ref,
        omega="idzorek",
        view_confidences=confidence_on_ref,
        tau=tau
    )

    mu_post = bl_model.bl_returns()
    sigma_post = bl_model.bl_cov()

    ef = EfficientFrontier(mu_post, sigma_post)
    if use_non_convex:
        weights = ef.nonconvex_objective(
            sharpe_with_l2,
            objective_args=(ef.expected_returns, ef.cov_matrix, l2),
            weights_sum_to_one=True,
        )
        if isinstance(weights, dict):
            ef_weights = weights
        else:
            ef_weights = ef.clean_weights(weights)
    else:
        if l2 > 0:
            ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe()
        ef_weights = ef.clean_weights()

    bl_weights = bl_model.bl_weights(risk_aversion=1)

    return {"Posterior mean": mu_post,
            "Posterior covariance": sigma_post,
            "Optimal Portfolio Weights BL": bl_weights,
            "Optimal Portfolio Weights EF": ef_weights}


# uses the above functions
def rolling_daily_black_litterman(price_df: pd.DataFrame, universe_tickers: list, view_tickers: list, 
                                  start_date: str, end_date: str, preds: np.ndarray, uncertainties: np.ndarray,
                                  market_caps,  market_data, boost = 0.0,
                                  tau: float = 0.025, use_prev_prior_mean: bool = False, 
                                  use_prev_prior_cov: bool = False, threshold: float = 0.01,
                                  use_non_convex: bool = False, l2: float = 0.1) -> dict:
    """
    Apply the daily Black–Litterman model over a rolling date range.

    Parameters:
        price_df (pd.DataFrame): Price data for the full universe (indexed by date).
        universe_tickers (list): List of tickers used for the full universe.
        view_tickers (list): List of tickers for which views are specified.
        start_date (str): Start date for the rolling window.
        end_date (str): End date for the rolling window.
        preds (np.ndarray): 2D array of predictions with shape (n_days, len(view_tickers)).
        uncertainties (np.ndarray): 2D array of uncertainties with shape (n_days, len(view_tickers)).
        tau (float): Weight on the views, defaults to 0.025.
        use_prev_prior_mean (bool): Whether to use previous posterior returns as the prior.
        use_prev_prior_cov (bool): Whether to use previous posterior covariance as the prior.
        threshold (float): Threshold to convert discrete predictions to absolute views.

    Returns:
        dict: Dictionary with dates as keys and the corresponding Black-Litterman results as values.
    
    Raises:
        AssertionError: If any view ticker is not in the universe tickers.
    """
    # Ensure that view tickers are a subset of the full universe
    assert set(view_tickers).issubset(set(universe_tickers)), "All view tickers must be in the universe tickers."
    # use as input
    #market_caps = get_market_caps(price_df, universe_tickers)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # convert preds and uncertainties to dfs
    if preds is None or uncertainties is None:
        views_df = None
        confidence_df = None
    else:
        views_df = thresholds_to_views(preds, threshold, view_tickers, start_date, end_date)
        confidence_df = uq_to_confidence(uncertainties, view_tickers, start_date, end_date, boost)
    
    # Download market portfolio data (e.g. SPY)
    #use as input
    #market_data = download_market_data(start_date, end_date)
    
    results = {}
    prev_posterior = None
    prev_cov = None
    
    for day in date_range:
        if day not in price_df.index:
            continue
        
        result = daily_black_litterman(
            price_df=price_df,
            ref_date=day,
            views=views_df,
            confidence=confidence_df,
            market_caps=market_caps,
            tau=tau,
            market_data=market_data,
            use_prev_prior_mean=use_prev_prior_mean,
            use_prev_prior_cov=use_prev_prior_cov,
            prev_posterior=prev_posterior,
            prev_cov=prev_cov,
            use_non_convex=use_non_convex,
            l2=l2
        )
        
        prev_posterior = result["Posterior mean"]
        prev_cov = result["Posterior covariance"]
        
        results[day.strftime('%Y-%m-%d')] = result
    
    return results


def simulate_portfolio(initial_wealth, weights, price_df):
    """
    Simulate the portfolio evolution given an initial wealth, daily rebalanced weights,
    and asset prices.

    Parameters:
        initial_wealth (float): The starting portfolio value.
        weights (pd.Series) : series with dates as index and portfolio weights, e.g., {"AAPL": 0.3, "GOOG": 0.2, ...} as values.
        price_df (pd.DataFrame): DataFrame of asset prices with dates as index and tickers as columns.

    Returns:
        pd.Series: Series of portfolio values over time (indexed by trading dates).
    """
    portfolio_values = {}
    rebalancing_dates = sorted(weights.index)
    if not rebalancing_dates:
        raise ValueError("weights_dict is empty.")

    # Initialize portfolio value on the first rebalancing date.
    current_wealth = initial_wealth
    portfolio_values[rebalancing_dates[0]] = current_wealth

    # Iterate over all dates and use continue in case the prices have the same number of dates
    for i in range(len(rebalancing_dates)-1):
        current_date = rebalancing_dates[i]
        next_date = rebalancing_dates[i + 1]

        if current_date not in price_df.index or next_date not in price_df.index:
            continue

        current_weights = weights[current_date] #returns a dictionary of structure {"AAPL": 0.3, "GOOG": 0.2, ...}
        weights_series = pd.Series(current_weights)
        
        # Extract asset prices for current and next day.
        current_prices = price_df.loc[current_date]
        next_prices = price_df.loc[next_date]
        
        # Calculate daily asset returns.
        asset_returns = (next_prices / current_prices) - 1
        # pcik the corresponding returns for the weights
        portfolio_return = (weights_series * asset_returns[weights_series.index]).sum()
        
        # Update portfolio wealth.
        current_wealth *= (1 + portfolio_return)
        portfolio_values[next_date] = current_wealth

    return pd.Series(portfolio_values)