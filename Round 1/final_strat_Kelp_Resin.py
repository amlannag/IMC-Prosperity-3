from typing import Dict, List, Any, Tuple, Optional, Union
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import numpy as np
import jsonpickle
import pandas as pd
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

class ARIMA:
    """
    A simplified implementation of ARIMA (AutoRegressive Integrated Moving Average) model
    using only the supported libraries.
    
    This implementation handles:
    - AR (AutoRegressive) component
    - I (Integration/Differencing)
    - MA (Moving Average) component
    """
    
    def __init__(self, order: Tuple[int, int, int]):
        """
        Initialize the ARIMA model with the specified order.
        
        Parameters:
        -----------
        order : Tuple[int, int, int]
            The (p, d, q) order of the model for the autoregressive, 
            differencing, and moving average components.
        """
        self.p, self.d, self.q = order
        self.ar_params = None
        self.ma_params = None
        self.intercept = None
        self.residuals = None
        self.mean = 0
        self.std = 1
        self.is_fitted = False
        
    def _difference(self, series: np.ndarray, d: int = 1) -> np.ndarray:
        """
        Apply differencing of order d to the time series.
        
        Parameters:
        -----------
        series : np.ndarray
            The time series data to difference
        d : int
            The differencing order
            
        Returns:
        --------
        np.ndarray
            The differenced time series
        """
        result = series.copy()
        for _ in range(d):
            result = np.diff(result)
        return result
    
    def _inverse_difference(self, differenced: np.ndarray, original: np.ndarray, d: int = 1) -> np.ndarray:
        """
        Invert the differencing operation to restore the original scale.
        
        Parameters:
        -----------
        differenced : np.ndarray
            The differenced time series
        original : np.ndarray
            The original time series before differencing
        d : int
            The differencing order
            
        Returns:
        --------
        np.ndarray
            The undifferenced time series
        """
        result = differenced
        for i in range(d):
            offset = len(original) - len(result) - 1
            result = np.r_[original[offset], result.cumsum() + original[offset]]
        return result
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model fitting by differencing and creating lagged values.
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series data
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The original data, differenced data, and lagged data for AR and MA
        """
        original_data = data.copy()
        
        # Apply differencing
        diff_data = self._difference(data, self.d)
        
        # Standardize the differenced data
        self.mean = np.mean(diff_data)
        self.std = np.std(diff_data)
        if self.std > 0:
            std_data = (diff_data - self.mean) / self.std
        else:
            std_data = diff_data - self.mean
            
        # Prepare lagged data for AR and MA components
        max_lag = max(self.p, self.q)
        if max_lag > 0:
            lagged_data = np.zeros((len(std_data) - max_lag, self.p + self.q))
            
            # Create lags for AR
            for i in range(self.p):
                lagged_data[:, i] = std_data[max_lag-i-1:len(std_data)-i-1]
            
            # Initial residuals are zero
            residuals = np.zeros(len(std_data))
            
            # Create lags for MA (initial pass with zeros)
            for i in range(self.q):
                lagged_data[:, self.p + i] = residuals[max_lag-i-1:len(std_data)-i-1]
                
            y = std_data[max_lag:]
        else:
            lagged_data = np.ones((len(std_data), 1))
            y = std_data
            
        return original_data, std_data, lagged_data, y
    
    def _ordinary_least_squares(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Simple OLS implementation to estimate model parameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix with lagged values
        y : np.ndarray
            Target values
            
        Returns:
        --------
        np.ndarray
            Estimated parameters
        """
        # Add a column of ones for the intercept
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Calculate parameters: β = (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX)
            params = XtX_inv @ X.T @ y
            return params
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for numerical stability if matrix is singular
            params = np.linalg.lstsq(X, y, rcond=None)[0]
            return params
    
    def fit(self, data: Union[pd.Series, np.ndarray, List[float]]) -> 'ARIMA':
        """
        Fit the ARIMA model to the data.
        
        Parameters:
        -----------
        data : Union[pd.Series, np.ndarray, List[float]]
            Time series data to fit the model
            
        Returns:
        --------
        ARIMA
            The fitted model
        """
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
            
        if len(data) <= max(self.p, self.d + self.q):
            raise ValueError("Data length is too short for the specified ARIMA order.")
            
        # Prepare data
        original_data, std_data, X, y = self._prepare_data(data)
        
        # Fit the model with OLS
        params = self._ordinary_least_squares(X, y)
        
        # Extract parameters
        self.intercept = params[0]
        if self.p > 0:
            self.ar_params = params[1:self.p+1]
        else:
            self.ar_params = np.array([])
            
        if self.q > 0:
            self.ma_params = params[self.p+1:self.p+self.q+1]
        else:
            self.ma_params = np.array([])
            
        # Calculate residuals
        predictions = self.intercept + X @ params[1:]
        self.residuals = y - predictions
        
        self.is_fitted = True
        return self
    

    def forecast(self, steps=1, return_conf_int=False, alpha=0.05, original_data=None):
        """
        Generate forecasts for future time steps.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast ahead
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals
        original_data : np.ndarray, optional
            Original data used to properly undifference forecasts
            
        Returns:
        --------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
            Forecasted values and optionally confidence intervals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forecasting.")
            
        # Initialize forecast array
        forecast = np.zeros(steps)
        
        # Initial residuals from model fitting
        residuals = self.residuals.copy()
        
        # Use last values from the original series for initial predictions
        ar_terms = np.zeros(self.p)
        ma_terms = np.zeros(self.q)
        
        # Set initial AR terms
        if self.p > 0 and len(residuals) >= self.p:
            ar_terms = residuals[-self.p:][::-1]
            
        # Set initial MA terms
        if self.q > 0 and len(residuals) >= self.q:
            ma_terms = residuals[-self.q:][::-1]
            
        # Generate forecasts
        for i in range(steps):
            # AR component
            ar_component = 0
            if self.p > 0:
                ar_component = np.sum(self.ar_params * ar_terms)
            
            # MA component
            ma_component = 0
            if self.q > 0:
                ma_component = np.sum(self.ma_params * ma_terms)
            
            # Calculate forecast
            yhat = self.intercept + ar_component + ma_component
            forecast[i] = yhat
            
            # Update terms for next prediction
            if self.p > 0:
                ar_terms = np.roll(ar_terms, 1)
                ar_terms[0] = yhat
                
            if self.q > 0:
                # For future steps, we assume residuals are 0
                ma_terms = np.roll(ma_terms, 1)
                ma_terms[0] = 0
        
        # Scale back to original scale (undo standardization)
        forecast = forecast * self.std + self.mean
        
        # Undifference the forecasts if differencing was applied
        if self.d > 0 and original_data is not None:
            # We need the last d values from the original data to undifference
            diff_forecasts = forecast.copy()
            last_values = original_data[-self.d:]
            
            # For each forecasted value
            for i in range(steps):
                # Add the most recent value to get back to original scale
                forecast[i] = diff_forecasts[i] + last_values[-1]
                # Update the most recent value for the next forecast
                last_values = np.append(last_values[1:], forecast[i])
        
        # Calculate confidence intervals if requested
        if return_conf_int:
            # Simple approximation of standard error
            std_err = np.std(self.residuals) * np.sqrt(np.arange(1, steps + 1))
            z_val = abs(np.percentile(np.random.standard_normal(10000), 100 * (1 - alpha/2)))
            
            lower_bound = forecast - z_val * std_err
            upper_bound = forecast + z_val * std_err
            
            return forecast, lower_bound, upper_bound
        
        return forecast

    
    def predict(self, data: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        """
        Generate one-step ahead predictions for the given data.
        
        Parameters:
        -----------
        data : Union[pd.Series, np.ndarray, List[float]]
            Time series data to predict
            
        Returns:
        --------
        np.ndarray
            One-step ahead predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
            
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
            
        # Apply differencing
        diff_data = self._difference(data, self.d)
        
        # Standardize
        std_data = (diff_data - self.mean) / self.std if self.std > 0 else diff_data - self.mean
        
        # Initialize predictions
        n = len(std_data)
        predictions = np.zeros(n)
        residuals = np.zeros(n)
        
        # Generate predictions
        for t in range(max(self.p, self.q), n):
            # AR component
            ar_component = 0
            if self.p > 0:
                ar_terms = std_data[t-self.p:t][::-1]
                ar_component = np.sum(self.ar_params * ar_terms)
            
            # MA component
            ma_component = 0
            if self.q > 0:
                ma_terms = residuals[t-self.q:t][::-1]
                ma_component = np.sum(self.ma_params * ma_terms)
            
            # Calculate prediction
            yhat = self.intercept + ar_component + ma_component
            predictions[t] = yhat
            
            # Update residuals
            residuals[t] = std_data[t] - yhat
        
        # Scale back to original scale
        predictions = predictions * self.std + self.mean
        
        # Handle initial values
        predictions[:max(self.p, self.q)] = np.nan
        
        return predictions
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file using jsonpickle.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'w') as f:
            f.write(jsonpickle.encode(self))
            
    @classmethod
    def load(cls, filepath: str) -> 'ARIMA':
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
            
        Returns:
        --------
        ARIMA
            The loaded model
        """
        with open(filepath, 'r') as f:
            return jsonpickle.decode(f.read())
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"ARIMA(order=({self.p},{self.d},{self.q}))"
    
    def summary(self) -> str:
        """
        Return a summary of the model.
        
        Returns:
        --------
        str
            Model summary
        """
        if not self.is_fitted:
            return "Model has not been fitted yet."
        
        summary = f"ARIMA Model: ({self.p}, {self.d}, {self.q})\n"
        summary += "=========================\n"
        summary += f"Intercept: {self.intercept:.4f}\n\n"
        
        if self.p > 0:
            summary += "AR Coefficients:\n"
            for i, param in enumerate(self.ar_params):
                summary += f"  AR[{i+1}]: {param:.4f}\n"
            summary += "\n"
            
        if self.q > 0:
            summary += "MA Coefficients:\n"
            for i, param in enumerate(self.ma_params):
                summary += f"  MA[{i+1}]: {param:.4f}\n"
            summary += "\n"
            
        # Basic model statistics
        if self.residuals is not None:
            residuals = self.residuals
            summary += "Model Statistics:\n"
            summary += f"  Std. Error: {np.std(residuals):.4f}\n"
            summary += f"  Log Likelihood: {-0.5 * len(residuals) * (1 + np.log(2 * np.pi * np.var(residuals))):.4f}\n"
            summary += f"  AIC: {2 * (self.p + self.q + 1) - 2 * (-0.5 * len(residuals) * (1 + np.log(2 * np.pi * np.var(residuals)))):.4f}\n"
            
        return summary

logger = Logger()

class Trader:
    def __init__(self):
        # Parameters for products
        self.rainforest_fair_value = 10000  # Fixed fair value for Rainforest Resin (like Amethysts)
        self.z_score_threshold = 2.5
        
        # Product-specific parameters
        self.params = {
            "RAINFOREST_RESIN": {
                "fair_value": 10000,  # Fixed value like Amethysts
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 10,
                "position_limit": 50
            },
            "KELP": {
                "take_width": 1,
                "clear_width": 0,
                "prevent_adverse": True,
                "adverse_volume": 15,
                "reversion_beta": -0.229,
                "disregard_edge": 1,
                "join_edge": 0,
                "default_edge": 1,
                "position_limit": 50
            },
            "SQUID_INK":{
                "take_width": 1,
                "clear_width": 0,
                "prevent_adverse": True,
                "adverse_volume": 15,
                "reversion_beta": -0.1781,
                "disregard_edge": 1,
                "join_edge": 0,
                "default_edge": 1,
                "position_limit": 50
            }
        }

    def calculate_vwap(self, orders: Dict[int, int]) -> float:
        """Calculate volume-weighted average price"""
        if not orders:
            return 0
        return sum(price * abs(vol) for price, vol in orders.items()) / sum(abs(vol) for vol in orders.values())

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int = 0,
        sell_order_volume: int = 0,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.params[product]["position_limit"]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int = 0,
        sell_order_volume: int = 0,
    ) -> (int, int):
        position_limit = self.params[product]["position_limit"]
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int = 0,
        sell_order_volume: int = 0,
    ) -> (int, int):
        position_limit = self.params[product]["position_limit"]
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int = 0,
        sell_order_volume: int = 0,
        disregard_edge: float = 1,  # disregard trades within this edge for pennying or joining
        join_edge: float = 2,  # join trades within this edge
        default_edge: float = 4,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        """Calculates fair value for KELP using the same method as STARFRUIT"""
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params["KELP"]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params["KELP"]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params["KELP"]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None
    def ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        """Calculates fair value for KELP using the same method as STARFRUIT"""
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params["SQUID_INK"]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params["SQUID_INK"]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            if mm_ask is None or mm_bid is None:
                if traderObject.get("ink_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("ink_last_price", None) is not None:
                last_price = traderObject["ink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params["SQUID_INK"]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            
            traderObject["ink_last_price"] = mmmid_price
            return fair
        return None

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Trading strategy implementation applying Amethysts strategy to Rainforest
        and Starfruit strategy to Kelp
        """
        # Initialize the method output dict as an empty dict
        result = {}
        
        # Initialize or load trader object
        if not state.traderData:
            traderObject = {
                "historical_data": {
                    "RAINFOREST_RESIN": [],
                    "KELP": [],
                    "SQUID_INK": []
                }
            }
        else:
            traderObject = jsonpickle.decode(state.traderData)
            if "historical_data" not in traderObject:
                traderObject["historical_data"] = {
                    "RAINFOREST_RESIN": [],
                    "KELP": [],
                    "SQUID_INK": []
                }
        
        # Ensure product data exists
        for product in state.order_depths.keys():
            if product not in traderObject["historical_data"]:
                traderObject["historical_data"][product] = []
        
        # Iterate over all products
        for product in state.order_depths.keys():
            # Get the order depth for this product
            order_depth = state.order_depths[product]
            
            # Get current position
            position = state.position.get(product, 0)
            
            # Initialize empty orders list
            orders: List[Order] = []
            
            # Apply Amethysts strategy to Rainforest_Resin
            if product == "RAINFOREST_RESIN":
                # Only proceed if we have order depth
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    # Calculate mid price for historical tracking
                    best_sell = min(order_depth.sell_orders.keys())
                    best_buy = max(order_depth.buy_orders.keys())
                    mid_price = (best_sell + best_buy) / 2
                    
                    # Take orders
                    buy_order_volume, sell_order_volume = 0, 0
                    rainforest_take_orders = []
                    buy_order_volume, sell_order_volume = self.take_best_orders(
                        product,
                        self.params[product]["fair_value"],
                        self.params[product]["take_width"],
                        rainforest_take_orders,
                        order_depth,
                        position,
                        buy_order_volume,
                        sell_order_volume
                    )
                    
                    # Clear orders
                    rainforest_clear_orders = []
                    buy_order_volume, sell_order_volume = self.clear_position_order(
                        product,
                        self.params[product]["fair_value"],
                        self.params[product]["clear_width"],
                        rainforest_clear_orders,
                        order_depth,
                        position,
                        buy_order_volume,
                        sell_order_volume
                    )
                    
                    # Make orders
                    rainforest_make_orders, _, _ = self.make_orders(
                        product,
                        order_depth,
                        self.params[product]["fair_value"],
                        position,
                        buy_order_volume,
                        sell_order_volume,
                        self.params[product]["disregard_edge"],
                        self.params[product]["join_edge"],
                        self.params[product]["default_edge"],
                        True,
                        self.params[product]["soft_position_limit"]
                    )
                    
                    orders = rainforest_take_orders + rainforest_clear_orders + rainforest_make_orders
                    
                    # Store mid price in historical data
                    traderObject["historical_data"][product].append(mid_price)
                    


            # Apply Starfruit strategy to Kelp
            elif product == "KELP":
                # # Only proceed if we have order depth
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    # Calculate mid price for historical tracking
                    best_sell = min(order_depth.sell_orders.keys())
                    best_buy = max(order_depth.buy_orders.keys())
                    mid_price = (best_sell + best_buy) / 2
                    
                    # Calculate fair value using the Starfruit approach
                    kelp_fair_value = self.kelp_fair_value(order_depth, traderObject)
                    
                    if kelp_fair_value is not None:
                        # Take orders
                        buy_order_volume, sell_order_volume = 0, 0
                        kelp_take_orders = []
                        buy_order_volume, sell_order_volume = self.take_best_orders(
                            product,
                            kelp_fair_value,
                            self.params[product]["take_width"],
                            kelp_take_orders,
                            order_depth,
                            position,
                            buy_order_volume,
                            sell_order_volume,
                            self.params[product]["prevent_adverse"],
                            self.params[product]["adverse_volume"]
                        )
                        
                        # Clear orders
                        kelp_clear_orders = []
                        buy_order_volume, sell_order_volume = self.clear_position_order(
                            product,
                            kelp_fair_value,
                            self.params[product]["clear_width"],
                            kelp_clear_orders,
                            order_depth,
                            position,
                            buy_order_volume,
                            sell_order_volume
                        )
                        
                        # Make orders
                        kelp_make_orders, _, _ = self.make_orders(
                            product,
                            order_depth,
                            kelp_fair_value,
                            position,
                            buy_order_volume,
                            sell_order_volume,
                            self.params[product]["disregard_edge"],
                            self.params[product]["join_edge"],
                            self.params[product]["default_edge"]
                        )
                        
                        orders = kelp_take_orders + kelp_clear_orders + kelp_make_orders
                    
                    # Store mid price in historical data
                    traderObject["historical_data"][product].append(mid_price)
                
                
            
            # elif product == "SQUID_INK":
            #     if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:

            #         position = state.position.get(product, 0)
            #         position_limit = 50  # Same for both products
                    
            #         # Calculate remaining position capacity
            #         buy_capacity = position_limit - position
            #         sell_capacity = position_limit + position
            #         squid_std = 68
            #         squid_window = 100
            #         z_boundry = 2.5
            #         z_score = 0.0

            #         buy_price = self.calculate_vwap(order_depth.buy_orders)
            #         sell_price = self.calculate_vwap(order_depth.sell_orders)
            #         mid_price = (buy_price + sell_price) / 2

            #         ink_fair_value = self.ink_fair_value(order_depth, traderObject)
                    
            #         prices = traderObject["historical_data"][product][-squid_window:]

            #         logger.print("PRICES",len(prices))

            #         if len(prices) >= squid_window:
            #             squid_std = np.std(prices)
            #             squid_mean = np.mean(prices)
                    
            #             z_score = ((mid_price - squid_mean)) / squid_std

            #         spread = squid_std

            #         inventory_factor = 0.00

            #         buy_price = int(mid_price - spread)
            #         sell_price = int(mid_price + spread)


            #         buy_volume = min(25,buy_capacity)
            #         sell_volume = min(25,sell_capacity)

            #         logger.print("Z_score",z_score)

            #         if z_score <= z_boundry and z_score >= -z_boundry:
                        
            #             if position == 50:
            #                 buy_price = int(mid_price - spread - inventory_factor * position)
            #                 sell_price = min(int(sell_price),int(mid_price + spread + inventory_factor * position))
            #                 orders.append(Order(product, sell_price, -50))
            #             elif position == -50:
            #                 buy_price = max(int(buy_price),int(mid_price - spread + inventory_factor * abs(position)))
            #                 sell_price = int(mid_price + spread - inventory_factor * abs(position))
            #                 orders.append(Order(product, buy_price, 50))


            #         elif z_score > z_boundry:
            #             orders.append(Order(product, sell_price, -50))

            #         elif z_score < -z_boundry:
            #             orders.append(Order(product, buy_price, 50))
                        
            #     result[product] = orders
            #     traderObject["historical_data"][product].append(mid_price)
                    
            elif product == "SQUID_INK":
                # if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                #     # Get position and calculate capacity
                #     position = state.position.get(product, 0)
                #     position_limit = 50
                    
                #     # Calculate mid price
                #     best_sell = min(order_depth.sell_orders.keys())
                #     best_buy = max(order_depth.buy_orders.keys())
                #     mid_price = (best_sell + best_buy) / 2
                    
                #     # Store current price in history
                #     traderObject["historical_data"][product].append(mid_price)
                    
                #     # Get the last price if available
                #     if len(traderObject["historical_data"][product]) > 21:
                #         last_price = traderObject["historical_data"][product][-10]
                        
                #         # Calculate percentage change
                #         percent_change = ((mid_price - last_price) / last_price) * 100
                        
                #         # Simple threshold-based strategy
                #         threshold = 2 # 2% change threshold
                        
                #         logger.print(f"SQUID_INK: Current price: {mid_price}, Last price: {last_price}")
                #         logger.print(f"Percent change: {percent_change:.2f}%")
                        
                #         # Clear existing orders
                #         orders = []
                        
                #         if percent_change > threshold:
                #             # If price increased significantly, sell everything
                #             if position > 0:
                #                 sell_quantity = position_limit - position
                #                 orders.append(Order(product, best_buy, -sell_quantity))
                #                 logger.print(f"SELLING {position} at {best_buy} - Price change too high")
                            
                #         elif percent_change < -threshold:
                #             # If price decreased significantly, buy everything
                #             if position < position_limit:
                #                 buy_quantity = position_limit - position
                #                 orders.append(Order(product, best_sell, buy_quantity))
                #                 logger.print(f"BUYING {buy_quantity} at {best_sell} - Price change too low")

                #         if position > 0:
                #             last_price = traderObject["historical_data"][product][-1]
                #             percent_change = ((mid_price - last_price) / last_price) * 100
                #             if percent_change < -3:
                #                 # If price is lower than last price, sell
                #                 orders.append(Order(product, best_buy, -position))
                #                 logger.print(f"SELLING {position} at {best_buy} - Price decreased")
                #         if position < 0:
                #             last_price = traderObject["historical_data"][product][-1]
                #             percent_change = ((mid_price - last_price) / last_price) * 100
                #             if percent_change > 3:
                #                 # If price is higher than last price, buy
                #                 orders.append(Order(product, best_sell, -position))
                #                 logger.print(f"BUYING {abs(position)} at {best_sell} - Price increased")
                    
                #     result[product] = orders
                continue
            
            # Add final orders to result
            if orders:
                result[product] = orders
        
        # Trim historical data if it gets too long
        for product in traderObject["historical_data"].keys():
            if len(traderObject["historical_data"][product]) > 70:
                traderObject["historical_data"][product] = traderObject["historical_data"][product][-50:]
        

        logger.flush(state, result, 0, state.traderData)
        # Update traderData
        state.traderData = jsonpickle.encode(traderObject)
        
        return result, 0, state.traderData