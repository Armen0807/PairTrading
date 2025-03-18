import datetime as dt
from idlelib.configdialog import tracers

import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm

from pydantic import field_validator
from loguru import logger
from typing import Dict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, Extra

from grt_lib_price_loader import Instrument
from inar_strat_types import AbstractStratConfig, StratHistory, to_strat_history
from grt_lib_orchestrator import AbstractBacktestStrategy
from grt_lib_order_book import Trade
import uuid


class PairStrategyTrade(Trade):
	def __init__(self):



class PairTradingConfig(AbstractStratConfig):
	class Config:
		arbitrary_types_allowed = True

		class UnderlyingDict(TypedDict):
			base_date: dt.date
			base_level: float
			formation_period: int
			k : int


	# useful when parsing Instrument in risky_asset for example
	# @field_validator('risky_asset', mode="before")
	# def get_instrument(cls, v):
	# 	if isinstance(v, list):
	# 		return [ins if isinstance(ins, Instrument) else Instrument.from_ric(ins) for ins in v]
	# 	return [v if isinstance(v, Instrument) else Instrument.from_ric(v)]

class PairTradingCalculus(StratHistory):
	normalized_prices : Dict[str, Dict[dt.date, float]] = {}
	distance_measure :Dict[str, Dict[dt.date, float]] = {}
	mean :Dict[str, Dict[dt.date, float]] = {}
	var:Dict[str, Dict[dt.date, float]] = {}
	covar :Dict[str, Dict[dt.date, float]] = {}
	beta :Dict[str, Dict[dt.date, float]] = {}
	resid :Dict[str, Dict[dt.date, float]] = {}
	delta_resid :Dict[str, Dict[dt.date, float]] = {}
	fuller_test :Dict[str, Dict[dt.date, float]] = {}
	decision :Dict[str, Dict[dt.date, float]] = {}
	spread :Dict[str, Dict[dt.date, float]] = {}
	mean_spread :Dict[str, Dict[dt.date, float]] = {}
	var_spread :Dict[str, Dict[dt.date, float]] = {}
	threshold :Dict[str, Dict[dt.date, float]] = {}

class StrategySignals(StratHistory):
	Entry :Dict[dt.date, float] = {}
	Exit : Dict[dt.date, float] = {}






class PairTradingBackTest(StratHistory):
	price_changes : Dict[str, Dict[dt.date, float]] = {}
	weight_at_t : Dict[str, Dict[dt.date, float]] = {}
	returns : Dict[str, Dict[dt.date, float]] = {}
	sharpe_ratio : Dict[str, Dict[dt.date, float]] = {}
	cumul_returns : Dict[str, Dict[dt.date, float]] = {}

class PairTradingCost(StratHistory):
	cost_bid_ask :Dict[str, Dict[dt.date, float]] = {}
	fixed_fees :Dict[str, Dict[dt.date, float]] = {}
	cost_slippage :Dict[str, Dict[dt.date, float]] = {}
	mid_price :Dict[str, Dict[dt.date, float]] = {}
	cost_total :Dict[str, Dict[dt.date, float]] = {}
	net_returns :Dict[str, Dict[dt.date, float]] = {}

class PairTradingRisk(StratHistory):
	# to fill : Dict[dt.date, float] = {}



class PairTradingHistory(StratHistory):
	Pair_trading: PairTradingCalculus = PairTradingCalculus()
	strategy_signals: StrategySignals = StrategySignals()
	Back_test: PairTradingBackTest = PairTradingBackTest()
	Trading_cost : PairTradingCost = PairTradingCost()
	Risk_management: PairTradingRisk = PairTradingRisk()

class PairTradingBacktestStrategy(AbstractBacktestStrategy):

	config: PairTradingConfig
	history: PairTradingHistory

	"""Intermediate calculation Tools for strategy"""

	def get_underlying_spot_price(self, underlying: Instrument, t: dt.date) -> float:
		underlying_instrument = Instrument.from_ric(underlying)
		return self.price_loader.get_price(t, underlying_instrument)

	@to_strat_history("intermediate", lookup_attrs_count=2)
	def calc_normalized_prices(self,underlying:Instrument,t:dt.date):
		return (self.get_underlying_spot_price(underlying,t)/self.underlying.base_level

	@to_strat_history("intermediate", lookup_attrs_count=3)
	def calc_distance_measure(self, underlying_1: Instrument,underlying_2:Instrument, t: dt.date) -> float:
		calc = 0
		for i in range(self.underlying.formation_period):
			tmi = self.calendar.busday_add(t,-i)
			p1 = self.calc_normalized_prices(underlying_1, tmi)
			p2 = self.calc_normalized_prices(underlying_2, tmi)
			calc+=np.square(p1-p2)
		return np.sqrt(calc)

	def calc_mean(self,underlying:Instrument,t:dt.date)->float:
		n = self.underlying.formation_period
		sum_ = 0
		for i in range(n):
			tmi = self.calendar.busday_add(t,-i)
			p_i = self.calc_normalized_prices(underlying,tmi)
			sum_+=p_i
		return sum_/n

	def calc_var(self,underlying:Instrument,t:dt.date)->float:
		n = self.underlying.formation_period
		sum_ = 0
		for i in range(n):
			tmi = self.calendar.busday_add(t,-i)
			p = self.calc_normalized_prices(underlying,tmi)
			mu = self.calc_mean(underlying,tmi)
			sum_+=np.square(p-mu)
		return (1/n)*sum_

	def calc_cov(self,underlying_1:Instrument,underlying_2 :Instrument,t:dt.date)->float:
		n = self.underlying.formation_period
		sum_=0
		for i in range(n):
			t_i = self.calendar.busday_add(t,i)
			p_1_i = self.calc_normalized_prices(underlying_1,t)
			p_2_i = self.calc_normalized_prices(underlying_2,t)
			mu_1 = self.calc_mean(underlying_1,t_i)
			mu_2 = self.calc_mean(underlying_2,t_i)
			sum_ += (p_1_i-mu_1)*(p_2_i-mu_2)
		return (1/n)*sum_

	def calc_beta(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->float:
		cov = self.calc_cov(underlying_1,underlying_2,t)
		var = self.calc_var(underlying_2,t)
		return cov/var

	def calc_resid(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->float:
		beta = self.calc_beta(underlying_1,underlying_2,t)
		p_1 = self.calc_normalized_prices(underlying_1,t)
		p_2 = self.calc_normalized_prices(underlying_1,t)
		return 	p_1 - beta * p_2

	def calc_delta_resid(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->float:
		tm1 = self.calendar.busday_add(t,-1)
		resid_t = self.calc_resid(underlying_1,underlying_2,t)
		resid_tm1 = self.calc_resid(underlying_1,underlying_2,tm1)
		return resid_t - resid_tm1



	def ad_fuller_test(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->tuple[float, float, float]:
		n =self.underlying.formation_period
		residuals = []
		for i in range(n):
			tmi = self.calendar.busday_add(t,-i)
			resid_i = self.calc_resid(underlying_1,underlying_2,tmi)
			residuals.append(resid_i)

		result = adfuller(residuals, maxlag=None, autolag='AIC')
		adf_stat = result[0]  # Test statistic
		p_value = result[1]  # p-value
		critical_values = result[4]

		return adf_stat,p_value,critical_values

	def criteria_decision(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->bool:
		adf_test = self.ad_fuller_test(underlying_1,underlying_2,t)

		if adf_test[0] > adf_test[2]:
			return True   # stationary
		else:
			return False  # not stationary
		######################################
	def calc_spread(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->float:
		p_1 = self.calc_normalized_prices(underlying_1,t)
		p_2 = self.calc_normalized_prices(underlying_1,t)
		return p_1 - p_2

	def calc_mean_spread(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->float:
		n = self.underlying.formation_period
		total = 0
		for i in range(n):
			tmi = self.calendar.busday_add(t,-i)
			p_1_i = self.calc_normalized_prices(underlying_1,tmi)
			p_2_i = self.calc_normalized_prices(underlying_2,tmi)
			spread = p_1_i - p_2_i
			total += spread
		return total / n

	def calc_var_spread(self,underlying_1:Instrument,underlying_2: Instrument,t:dt.date)->float:
		n = self.underlying.formation_period
		sum_ = 0
		for i in range(n):
			tmi = self.calendar.busday_add(t, -i)
			p_1_i = self.calc_normalized_prices(underlying_1,tmi)
			p_2_i = self.calc_normalized_prices(underlying_1,tmi)
			spread = p_1_i - p_2_i
			mu = self.calc_mean_spread(underlying_1,underlying_2, tmi)
			sum_ += np.square(spread - mu)
		return (1 / n) * sum_

	def calc_threshold(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date, k : float,type_:str ="long"):
		mean = self.calc_mean_spread(underlying_1,underlying_2, t)
		vol =np.sqrt(self.calc_var_spread(underlying_1,underlying_2, t))
		if type_ == "long":
			return mean - k * vol
		elif type_ == "short":
			return mean + k * vol
		else:
			return TypeError("Please enter a correct value for type_, either long or short")

	"""Trading Strategy Section Tools"""


	def strategy(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date)->AbstractStratConfig:
		spread = self.calc_var_spread(underlying_1,underlying_2, t)
		mean_spread = self.calc_mean_spread(underlying_1,underlying_2, t)
		var_spread = self.calc_var_spread(underlying_1,underlying_2, t)
		k = self.underlying.threshold_parameter
		# long position entry conditions
		if spread > mean_spread + k *np.sqrt(var_spread):
			entry_price = self.get_underlying_spot_price(underlying_1,t)
			long = self.OrderBook.open_position("stock",entry_price )
		elif spread > mean_spread - k * np.sqrt(var_spread):



	"""backtest Section tools"""
	def get_number_of_shares(self,identifier:uuid.UUID)->int:
		trade = self.OrderBook.find_trade_by_identifier(identifier)
		return trade.number_of_shares()
	def get_entry_price(self,identifier:uuid.UUID)->float:
		trade = self.OrderBook.find_trade_by_identifier(identifier)
		return trade.entry_price()

	"""backtest Section tools"""

	def calc_price_change(self,underlying_1:Instrument,t:dt.date)->float:
		spot = self.get_underlying_spot_price(underlying_1,t)
		tm1 = self.calendar.busday_add(t,-1)
		spot_tm1 = self.calc_spot_price(underlying_1,tm1)
		return (spot - spot_tm1)/spot_tm1

	def calc_porfolio_value(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:

		number_1 = self.get_number_of_shares(identifier_1)
		number_2 = self.get_number_of_shares(identifier_2)
		spot_1 = self.get_underlying_spot_price(underlying_1, t)
		spot_2 = self.get_underlying_spot_price(underlying_2, t)
		value_1 = spot_1 * number_1
		value_2 = spot_2 * number_2
		return value_1 + value_2

	def calc_weight_at_t(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		number_1 = self.get_number_of_shares(identifier_1)
		number_2 = self.get_number_of_shares(identifier_2)
		spot_1 = self.get_underlying_spot_price(underlying_1,t)
		spot_2 = self.get_underlying_spot_price(underlying_2,t)
		value_1 = spot_1 * number_1
		value_2 = spot_2 * number_2
		return value_1 / (value_1 + value_2)

	def calc_returns(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		weight_1 = self.calc_weight_at_t(underlying_1,underlying_2,t,identifier_1,identifier_2)
		weight_2 = self.calc_weight_at_t(underlying_2,underlying_1,t,identifier_2,identifier_1)
		price_1 = self.calc_price_change(underlying_1,t)
		price_2 = self.calc_price_change(underlying_2,t)
		return weight_1 * price_1 + weight_2 * price_2

	def calc_cumul_returns(self,underlying_1:Instrument,underlying_2: Instrument, t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		ret = 1
		n = self.underlying.formation_period
		for i in range(n):
			tmi = self.calendar.busday_add(t,-i)
			daily_ret = self.calc_returns(underlying_1,underlying_2,tmi,identifier_1,identifier_2)
			ret*= (1+daily_ret)-1
		return ret

	def sharpe_ratio(self,underlying_1:Instrument,underlying_2: Instrument, t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID,risk_free_rate : float):
		daily_ret = self.calc_returns(underlying_1, underlying_2, t, identifier_1, identifier_2)
		mean_ret = daily_ret.mean()
		vol = daily_ret.std(ddof=0)
		return mean_ret - risk_free_rate / vol

	"""Transaction Cost Section Tools"""
	def calc_cost_bid_ask(self,instrument_ric:str,t:dt.date):
		instrument = Instrument.from_ric(instrument_ric)
		bid_price = self.price_loader.get_price(t,instrument,"Bid")
		ask_price = self.price_loader.get_price(t,instrument,"Ask")
		spread = ask_price - bid_price
		mid_price = (ask_price + bid_price)/2
		return spread/mid_price

	def calc_fixed_fees(self,identifier :uuid.UUID )->float:
		trade = self.OrderBook.find_trade_by_identifier(identifier)
		broker_fees = trade.broker_fees()
		number_of_shares = self.get_number_of_shares(identifier)

		return broker_fees * number_of_shares

	def calc_mid_price(self,instrument_ric:str,t:dt.date):
		instrument = Instrument.from_ric(instrument_ric)
		bid_price = self.price_loader.get_price(t, instrument, "Bid")
		ask_price = self.price_loader.get_price(t, instrument, "Ask")
		return (ask_price + bid_price) / 2

	def calc_cost_slippage(self,instrument_ric:str,t:dt.date,identifier :uuid.UUID):
		instrument = Instrument.from_ric(instrument_ric)
		mid_price = self.calc_mid_price(instrument,t)
		execution_price = self.get_entry_price(identifier)         ########### pas clair
		return abs(execution_price - mid_price)/mid_price

	def calc_cost_total(self,instrument_ric:str,t:dt.date,identifier :uuid.UUID):
		instrument = Instrument.from_ric(instrument_ric)
		fixed_fees = self.calc_fixed_fees(identifier)
		slippage = self.calc_cost_slippage(instrument_ric,t,identifier)
		bid_ask_cost = self.calc_cost_bid_ask(instrument,t)
		return fixed_fees + slippage + bid_ask_cost

	def calc_net_returns(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		gross_rets = self.calc_returns(underlying_1,underlying_2,t,identifier_1,identifier_2)
		transaction_costs = self.calc_cost_total(underlying_1,t,identifier_1)
		return gross_rets-transaction_costs

	"""Risk Management Section Tools"""
	def calc_VaR(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,confidence_level:float,period:int = 7)->float:
		cumul_rets = self.calc_cumulative_returns(underlying_1,underlying_2,t)
		std_ret = cumul_rets.std(ddof=0)
		z_score = norm.ppf(confidence_level)
		return -std_ret*z_score*np.srt(period)


	def calc_max_allowable_loss(self,risk_tolerance,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID):
		portfolio_value = self.calc_porfolio_value(underlying_1,underlying_2,t,identifier_1,identifier_2)
		return portfolio_value * risk_tolerance

	def calc_leverage_limits(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,confidence_level:float,period:int = 7,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		VaR = self.calc_VaR(underlying_1,underlying_2,t,confidence_level,period)
		max_loss = self.calc_max_allowable_loss(VaR,underlying_1,underlying_2,t,identifier_1,identifier_2)
		return VaR/max_loss

	def calc_equity(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		gross_portfolio_value = self.portfolio_value(underlying_1,underlying_2,t,identifier_1,identifier_2)
		liabilities = self.#get liabilities
		return gross_portfolio_value - liabilities

	def calc_portfolio_leverage(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID):
		gross_portfolio_value = self.calc_porfolio_value(underlying_1,underlying_2,t,identifier_1,identifier_2)
		equity = self.calc_equity(underlying_1,underlying_2,t,identifier_1,identifier_2)
		return gross_portfolio_value/equity

	def calc_adjusted_exposure(self,underlying_1:Instrument,underlying_2:Instrument,t:dt.date,identifier:uuid.UUID,confidence_level:float,period:int = 7)->float:
		trade = self.OrderBook.find_trade_by_identifier(identifier)
		VaR = self.calc_VaR(underlying_1,underlying_2,t,confidence_level,period)
		current_exposure = trade.current_exposure()
		max_allowable_exposure = self.calc_max_allowable_exposure(trade,identifier,t)
		return (max_allowable_exposure/VaR)*current_exposure

	def calc_stop_loss_VaR(self,underlying_1:Instrument,underlying_2:Instrument,identifier_1:uuid.UUID,identifier_2:uuid.UUID,t:dt.date,confidence_level:float,period:int = 7)->float:
		VaR = self.calc_VaR(underlying_1,underlying_2,t,confidence_level)
		portfolio_value = self.calc_porfolio_value(underlying_1,underlying_2,t,identifier_1,identifier_2)
		return portfolio_value * VaR

	def calc_max_cumul_returns(self,underlying_1:Instrument,underlying_2: Instrument, t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		n = self.underlying.formation_period
		highest = 0
		for i in range(n):
			tmi = self.calendar.busday_add(t,-i)
			cumul = self.calc_cumul_returns(underlying_1,underlying_2,tmi,identifier_1,identifier_2)
			if cumul > highest:
				highest = cumul
		return highest

	def calc_drawdown(self,underlying_1:Instrument,underlying_2: Instrument, t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		max_cumul = self.calc_max_cumul_returns(underlying_1,underlying_2,t,identifier_1,identifier_2)
		cumul = self.calc_cumul_returns(underlying_1,underlying_2,t,identifier_1,identifier_2)
		return 1-(cumul/max_cumul)

	def calc_max_drawdown(self,underlying_1:Instrument,underlying_2: Instrument, t:dt.date,identifier_1:uuid.UUID,identifier_2:uuid.UUID)->float:
		maximum = 0
		n = self.underlying.formation_period
		for i in range(n):
			tmi = self.calendar.busday_add(t,-i)
			drawdown = self.calc_drawdown(underlying_1,underlying_2,tmi,identifier_1,identifier_2)
			if drawdown > maximum:
				maximum = drawdown
		return maximum

	def calc_max_divergence(self,):
		#flemme de zinzin
