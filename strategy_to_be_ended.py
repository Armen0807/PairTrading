import datetime as dt
from typing import Dict, Optional, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm

from pydantic import BaseModel, Field

from nautilus_trader.common.instruments import Instrument, Quantity
from nautilus_trader.common.enums import OrderSide, OrderType, PositionSide
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.trading.execution_commands import LimitOrderCommand, MarketOrderCommand
from nautilus_trader.trading.position import Position
from nautilus_trader.common.timeframes import Timeframe
from nautilus_trader.common.periods import Period
from nautilus_trader.common.frequency import Frequency
from nautilus_trader.common.component import Component

class PairTradingConfig(BaseModel):
    underlying_1: str = Field("AAPL.US", description="Ticker of the first underlying asset")
    underlying_2: str = Field("SPY.US", description="Ticker of the second underlying asset")
    base_date: dt.date = Field(dt.date(2023, 1, 1), description="Date for price normalization")
    base_level: float = Field(100.0, description="Base level for price normalization")
    formation_period: int = Field(252, description="Number of days for the formation period")
    entry_threshold_std: float = Field(2.0, description="Entry threshold for residuals (in standard deviations)")
    exit_threshold_abs: float = Field(0.5, description="Exit threshold for absolute residual value")
    max_holding_days: int = Field(30, description="Maximum number of days to hold a position")
    residual_window: int = Field(20, description="Window size for calculating residual mean and standard deviation")
    beta_window: int = Field(60, description="Window size for calculating dynamic beta")
    transaction_cost_per_share: float = Field(0.01, description="Transaction cost per share")
    profit_target_multiplier: Optional[float] = Field(1.0, description="Multiplier for setting profit target based on entry residual")
    stop_loss_multiplier: Optional[float] = Field(1.0, description="Multiplier for setting stop loss based on entry residual")
    position_size: int = Field(100, description="Number of shares to trade")
    trailing_stop_loss_atr_period: int = Field(14, description="Period for ATR calculation for trailing stop loss")
    trailing_stop_loss_atr_multiplier: float = Field(2.0, description="Multiplier for ATR to set trailing stop loss")
    risk_free_rate: float = Field(0.02, description="Annual risk-free rate for Sharpe ratio calculation")
    volatility_lookback_days: int = Field(90, description="Lookback window for volatility calculation")
    adf_critical_level: float = Field(0.05, description="P-value threshold for ADF test stationarity")
    lag_correlation_window: int = Field(30, description="Window for lagged correlation analysis")
    max_leverage: float = Field(2.0, description="Maximum allowable leverage")
    initial_capital: float = Field(100000.0, description="Initial capital for backtesting")

class PairTradingStrategy(Strategy):
    config: PairTradingConfig
    instrument1: Instrument
    instrument2: Instrument
    normalized_prices: Dict[dt.date, Dict[str, float]] = {}
    spreads: Dict[dt.date, float] = {}
    betas: Dict[dt.date, float] = {}
    residuals: Dict[dt.date, float] = {}
    position_entry_date: Optional[dt.date] = None
    entry_residual: Optional[float] = None
    trailing_stop_loss: Optional[float] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = PairTradingConfig(**self.config)
        self.instrument1 = self.engine.get_instrument(self.config.underlying_1)
        self.instrument2 = self.engine.get_instrument(self.config.underlying_2)
        if not self.instrument1 or not self.instrument2:
            raise ValueError("One or both instruments not found.")
        self.normalized_prices = {}
        self.spreads = {}
        self.betas = {}
        self.residuals = {}
        self.position_entry_date = None
        self.entry_residual = None
        self.trailing_stop_loss = None
        self.portfolio_value_history: Dict[dt.date, float] = {}
        self.returns_history: Dict[dt.date, float] = {}
        self.cumulative_returns_history: Dict[dt.date, float] = {}
        self.drawdown_history: Dict[dt.date, float] = {}
        self.max_drawdown = 0.0
        self.transaction_costs_history: Dict[dt.date, float] = {}
        self.total_profit = 0.0
        self.equity = self.config.initial_capital
        self.positions_history: Dict[dt.date, Optional[Position]] = {}

    def get_price(self, instrument: Instrument, bar: Bar) -> float:
        if bar and bar.instrument_id == instrument.instrument_id:
            return bar.close
        return np.nan

    def normalize_price(self, instrument: Instrument, date: dt.date) -> Optional[float]:
        history = self.engine.get_historical_data(
            instrument=instrument,
            timeframe=Timeframe.DAILY,
            period=Period.since(self.config.base_date),
        )
        df = pd.DataFrame([{'ts': bar.ts, 'close': bar.close} for bar in history])
        df['ts'] = pd.to_datetime(df['ts']).dt.date
        df = df.set_index('ts')
        if date in df.index:
            base_history = self.engine.get_historical_data(
                instrument=instrument,
                timeframe=Timeframe.DAILY,
                period=Period.since(self.config.base_date),
            )
            base_df = pd.DataFrame([{'ts': bar.ts, 'close': bar.close} for bar in base_history])
            base_df['ts'] = pd.to_datetime(base_df['ts']).dt.date
            base_df = base_df.set_index('ts')
            if self.config.base_date in base_df.index and date in df.index:
                base_price = base_df.loc[self.config.base_date, 'close']
                current_price = df.loc[date, 'close']
                normalized_price = (current_price / base_price) * self.config.base_level
                self.normalized_prices.setdefault(date, {})[instrument.symbol] = normalized_price
                return normalized_price
        return None

    def calculate_spread(self, date: dt.date) -> Optional[float]:
        price1 = self.normalize_price(self.instrument1, date)
        price2 = self.normalize_price(self.instrument2, date)
        if price1 is not None and price2 is not None:
            spread = price1 - price2
            self.spreads[date] = spread
            return spread
        return None

    def calculate_dynamic_beta(self, history_length: int) -> Optional[float]:
        dates = sorted(self.normalized_prices.keys())[-history_length:]
        if len(dates) < self.config.beta_window:
            return None
        prices1 = [self.normalized_prices[d].get(self.instrument1.symbol) for d in dates if self.normalized_prices[d].get(self.instrument1.symbol) is not None and self.normalized_prices[d].get(self.instrument2.symbol) is not None]
        prices2 = [self.normalized_prices[d].get(self.instrument2.symbol) for d in dates if self.normalized_prices[d].get(self.instrument1.symbol) is not None and self.normalized_prices[d].get(self.instrument2.symbol) is not None]

        if len(prices1) < self.config.beta_window or len(prices2) < self.config.beta_window:
            return None

        returns1 = np.diff(np.log(prices1[-self.config.beta_window:]))
        returns2 = np.diff(np.log(prices2[-self.config.beta_window:]))

        if len(returns2) > 0:
            beta, alpha = np.polyfit(returns2, returns1, 1)
            return beta
        return None

    def calculate_residual(self, date: dt.date) -> Optional[float]:
        beta = self.calculate_dynamic_beta(self.config.beta_window + self.config.residual_window)
        price1 = self.normalize_price(self.instrument1, date)
        price2 = self.normalize_price(self.instrument2, date)
        if beta is not None and price1 is not None and price2 is not None:
            residual = price1 - beta * price2
            self.residuals[date] = residual
            self.betas[date] = beta
            return residual
        return None

    def calculate_residual_zscore(self, date: dt.date) -> Optional[float]:
        residual_history = pd.Series(self.residuals).sort_index()
        residual_window_data = residual_history[residual_history.index <= date].last(f'{self.config.residual_window}D')
        if len(residual_window_data) < self.config.residual_window:
            return None
        mean_resid = residual_window_data.mean()
        std_resid = residual_window_data.std()
        current_residual = self.residuals.get(date)
        if current_residual is not None and std_resid != 0:
            return (current_residual - mean_resid) / std_resid
        return None

    def on_bar(self, bar: Bar):
        trade_date = bar.trade_date.date()
        self.calculate_spread(trade_date)
        residual = self.calculate_residual(trade_date)
        if residual is None:
            return

        zscore = self.calculate_residual_zscore(trade_date)
        if zscore is None:
            return

        current_position = self.get_position(self.instrument1)

        # Exit Logic
        if current_position:
            if self.position_entry_date and (trade_date - self.position_entry_date).days >= self.config.max_holding_days:
                self.close_position(self.instrument1)
                return

            if self.entry_residual is not None:
                if self.config.profit_target_multiplier is not None:
                    profit_target = self.entry_residual * self.config.profit_target_multiplier
                    if (self.entry_residual > 0 and residual <= profit_target) or (self.entry_residual < 0 and residual >= profit_target):
                        self.close_position(self.instrument1)
                        return
                if self.config.stop_loss_multiplier is not None:
                    stop_loss = self.entry_residual * self.config.stop_loss_multiplier * -1 # Stop loss is in the opposite direction
                    if (self.entry_residual > 0 and residual >= stop_loss) or (self.entry_residual < 0 and residual <= stop_loss):
                        self.close_position(self.instrument1)
                        return
                if self.config.exit_threshold_abs is not None and abs(residual) <= self.config.exit_threshold_abs:
                    self.close_position(self.instrument1)
                    return

                # Trailing Stop Loss
                if self.config.trailing_stop_loss_atr_multiplier > 0 and self.config.trailing_stop_loss_atr_period > 0:
                    history1 = self.engine.get_historical_data(instrument=self.instrument1, timeframe=Timeframe.DAILY, period=Period.since(trade_date - dt.timedelta(days=self.config.trailing_stop_loss_atr_period + 5)))
                    df_hist1 = pd.DataFrame([{'ts': h_bar.ts, 'high': h_bar.high, 'low': h_bar.low, 'close': h_bar.close} for h_bar in history1])
                    if not df_hist1.empty:
                        df_hist1['ts'] = pd.to_datetime(df_hist1['ts']).dt.date
                        df_hist1.set_index('ts', inplace=True)
                        df_hist1 = df_hist1[~df_hist1.index.duplicated(keep='last')]
                        df_hist1 = df_hist1.sort_index()
                        df_window = df_hist1.loc[:trade_date].tail(self.config.trailing_stop_loss_atr_period)
                        if len(df_window) == self.config.trailing_stop_loss_atr_period:
                            high = df_window['high'].max()
                            low = df_window['low'].min()
                            atr = (high - low) / 2 # Simple approximation
                            if current_position.side == PositionSide.LONG and self.trailing_stop_loss is not None:
                                self.trailing_stop_loss = max(self.trailing_stop_loss, bar.close - atr * self.config.trailing_stop_loss_atr_multiplier)
                                if bar.close <= self.trailing_stop_loss:
                                    self.close_position(self.instrument1)
                                    return
                            elif current_position.side == PositionSide.SHORT and self.trailing_stop_loss is not None:
                                self.trailing_stop_loss = min(self.trailing_stop_loss, bar.close + atr * self.config.trailing_stop_loss_atr_multiplier)
                                if bar.close >= self.trailing_stop_loss:
                                    self.close_position(self.instrument1)
                                    return


        # Entry Logic
        if not current_position:
            if zscore > self.config.entry_threshold_std:
                # Short the spread (short underlying_1, long underlying_2)
                quantity1 = Quantity.from_int(self.config.position_size * -1)
                quantity2 = Quantity.from_int(int(abs(self.config.position_size * self.betas.get(trade_date, 1.0))))
                if quantity2 > 0:
                    self.submit_order(LimitOrderCommand(self.instrument1, OrderSide.SELL, quantity1, bar.close))
                    self.submit_order(LimitOrderCommand(self.instrument2, OrderSide.BUY, quantity2, bar.close))
                    self.position_entry_date = trade_date
                    self.entry_residual = residual
                    # Initial trailing stop loss for short position
                    history1 = self.engine.get_historical_data(instrument=self.instrument1, timeframe=Timeframe.DAILY, period=Period.since(trade_date - dt.timedelta(days=self.config.trailing_stop_loss_atr_period + 5)))
                    df_hist1 = pd.DataFrame([{'ts': h_bar.ts, 'high': h_bar.high, 'low': h_bar.low, 'close': h_bar.close} for h_bar in history1])
                    if not df_hist1.empty:
                        df_hist1['ts'] = pd.to_datetime(df_hist1['ts']).dt.date
                        df_hist1.set_index('ts', inplace=True)
                        df_hist1 = df_hist1[~df_hist1.index.duplicated(keep='last')]
                        df_hist1 = df_hist1.sort_index()
                        df_window = df_hist1.loc[:trade_date].tail(self.config.trailing_stop_loss_atr_period)
                        if len(df_window) == self.config.trailing_stop_loss_atr_period:
                            low = df_window['low'].min()
                            atr = (df_window['high'].max() - low) / 2
                            self.trailing_stop_loss = bar.close + atr * self.config.trailing_stop_loss_atr_multiplier


            elif zscore < -self.config.entry_threshold_std:
                # Long the spread (long underlying_1, short underlying_2)
                quantity1 = Quantity.from_int(self.config.position_size)
                quantity2 = Quantity.from_int(int(abs(self.config.position_size * self.betas.get(trade_date, 1.0))) * -1)
                if quantity2 < 0:
                    self.submit_order(LimitOrderCommand(self.instrument1, OrderSide.BUY, quantity1, bar.close))
                    self.submit_order(LimitOrderCommand(self.instrument2, OrderSide.SELL, quantity2, bar.close))
                    self.position_entry_date = trade_date
                    self.entry_residual = residual
                    # Initial trailing stop loss for long position
                    history1 = self.engine.get_historical_data(instrument=self.instrument1, timeframe=Timeframe.DAILY, period=Period.since(trade_date - dt.timedelta(days=self.config.trailing_stop_loss_atr_period + 5)))
                    df_hist1 = pd.DataFrame([{'ts': h_bar.ts, 'high': h_bar.high, 'low': h_bar.low, 'close': h_bar.close} for h_bar in history1])
                    if not df_hist1.empty:
                        df_hist1['ts'] = pd.to_datetime(df_hist1['ts']).dt.date
                        df_hist1.set_index('ts', inplace=True)
                        df_hist1 = df_hist1[~df_hist1.index.duplicated(keep='last')]
                        df_hist1 = df_hist1.sort_index()
                        df_window = df_hist1.loc[:trade_date].tail(self.config.trailing_stop_loss_atr_period)
                        if len(df_window) == self.config.trailing_stop_loss_atr_period:
                            high = df_window['high'].max()
                            atr = (high - df_window['low'].min()) / 2
                            self.trailing_stop_loss = bar.close - atr * self.config.trailing_stop_loss_atr_multiplier

    def close_position(self, instrument: Instrument):
        position = self.get_position(instrument)
        if position:
            if position.side == PositionSide.LONG:
                self.submit_order(MarketOrderCommand(instrument, OrderSide.SELL, position.net_quantity))
            elif position.side == PositionSide.SHORT:
                self.submit_order(MarketOrderCommand(instrument, OrderSide.BUY, abs(position.net_quantity)))
            self.position_entry_date = None
            self.entry_residual = None
            self.trailing_stop_loss = None

    def on_trade(self, trade: Trade):
        self.positions_history[trade.trade_date.date()] = self.get_position(trade.instrument)
        cost = abs(trade.quantity) * self.config.transaction_cost_per_share
        self.transaction_costs_history.setdefault(trade.trade_date.date(), 0.0)
        self.transaction_costs_history[trade.trade_date.date()] += cost
        if self.position_entry_date is not None and trade.instrument == self.instrument1:
            pass # Track entry price and size if needed

    def on_bar_market_data(self, bar: Bar):
        trade_date = bar.trade_date.date()
        portfolio_value = self.calculate_portfolio_value(trade_date)
        self.portfolio_value_history[trade_date] = portfolio_value
        self.calculate_returns(trade_date, portfolio_value)
        self.calculate_cumulative_returns(trade_date)
        self.calculate_drawdown(trade_date)
        self.equity = portfolio_value - sum(t_cost for t_cost in self.transaction_costs_history.values())

    def calculate_portfolio_value(self, date: dt.date) -> float:
        value = self.config.initial_capital
        position1 = self.get_position(self.instrument1)
        position2 = self.get_position(self.instrument2)
        price1 = self.get_price_at_date(self.instrument1, date)
        price2 = self.get_price_at_date(self.instrument2, date)

        if position1 and price1 is not None:
            value += position1.net_quantity * price1
        if position2 and price2 is not None:
            value += position2.net_quantity * price2
        return value

    def get_price_at_date(self, instrument: Instrument, date: dt.date) -> Optional[float]:
        history = self.engine.get_historical_data(
            instrument=instrument,
            timeframe=Timeframe.DAILY,
            period=Period.since(self.config.base_date),
        )
        df = pd.DataFrame([{'ts': bar.ts, 'close': bar.close} for bar in history])
        df['ts'] = pd.to_datetime(df['ts']).dt.date
        df = df.set_index('ts')
        if date in df.index:
            return df.loc[date, 'close']
        return None

    def calculate_returns(self, date: dt.date, current_portfolio_value: float):
        previous_date = self.engine.time_handler.previous_business_day(date)
        previous_portfolio_value = self.portfolio_value_history.get(previous_date, self.config.initial_capital)
        if previous_portfolio_value != 0:
            returns = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
            self.returns_history[date] = returns

    def calculate_cumulative_returns(self, date: dt.date):
        cumulative_return = (self.portfolio_value_history[date] / self.config.initial_capital) - 1
        self.cumulative_returns_history[date] = cumulative_return

    def calculate_drawdown(self, date: dt.date):
        cumulative_returns = pd.Series(self.cumulative_returns_history).sort_index()
        if cumulative_returns.empty:
            return
        peak = cumulative_returns[:date].max() if date in cumulative_returns else 0
        current_return = cumulative_returns.get(date, 0)
        drawdown = peak - current_return
        self.drawdown_history[date] = drawdown
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def calculate_sharpe_ratio(self, lookback_days: int = 252) -> Optional[float]:
        returns_series = pd.Series(self.returns_history).sort_index().tail(lookback_days)
        if len(returns_series) < 2:
            return None
        mean_return = returns_series.mean()
        std_dev_return = returns_series.std()
        if std_dev_return == 0:
            return None
        annualized_mean_return = mean_return * 252
        annualized_std_dev_return = std_dev_return * np.sqrt(252)
        sharpe = (annualized_mean_return - self.config.risk_free_rate) / annualized_std_dev_return
        return sharpe

    def calculate_volatility(self, lookback_days: int = 90) -> Optional[float]:
        returns_series = pd.Series(self.returns_history).sort_index().tail(lookback_days)
        if len(returns_series) < 2:
            return None
        return returns_series.std() * np.sqrt(252)

    def perform_adf_test(self, lookback_days: int = 252) -> Optional[Tuple[float, float]]:
        spread_series = pd.Series(self.spreads).sort_index().tail(lookback_days).dropna()
        if len(spread_series) < self.config.formation_period:
            return None
        result = adfuller(spread_series, autolag='AIC')
        adf_statistic = result[0]
        p_value = result[1]
        return adf_statistic, p_value

    def calculate_lagged_correlation(self, lag: int = 1, lookback_days: int = 30) -> Optional[float]:
        prices1_series = pd.Series(self.normalized_prices.get(self.instrument1.symbol, {})).sort_index().tail(lookback_days).dropna()
        prices2_series = pd.Series(self.normalized_prices.get(self.instrument2.symbol, {})).sort_index().tail(lookback_days).dropna()
        if len(prices1_series) < lookback_days - abs(lag) or len(prices2_series) < lookback_days - abs(lag):
            return None
        if lag > 0:
            corr = prices1_series[:-lag].corr(prices2_series[lag:])
        elif lag < 0:
            corr = prices1_series[-lag:].corr(prices2_series[:lag])
        else:
            corr = prices1_series.corr(prices2_series)
        return corr

    def calculate_leverage(self, date: dt.date) -> float:
        portfolio_value = self.portfolio_value_history.get(date, self.config.initial_capital)
        equity = self.equity
        if equity != 0:
            return portfolio_value / equity
        return 0.0
