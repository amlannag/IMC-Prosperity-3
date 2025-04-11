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

logger = Logger()




class Trader:
    def __init__(self):
        # Set fair value for Rainforest Resin
        self.rainforest_fair_value = 10000  # Starting estimate
        self.z_score_threshold = 2

    def calculate_vwap(self, orders: Dict[int, int]) -> float:
        """Calculate volume-weighted average price"""
        if not orders:
            return 0
        return sum(price * abs(vol) for price, vol in orders.items()) / sum(abs(vol) for vol in orders.values())

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Trading strategy implementation with arbitrage for Rainforest Resin
        and a simpler approach for Kelp
        """
        # Initialize the method output dict as an empty dict
        result = {}
        current_data = {}
        
        # Initialize or load historical data from traderData
        if not state.traderData:
            historical_data = {
                "RAINFOREST_RESIN": [],
                "KELP": [],
                "SQUID_INK": []
            }
        else:
            historical_data = jsonpickle.decode(state.traderData)
            # Ensure all products exist in historical data
            for product in state.order_depths.keys():
                if product not in historical_data:
                    historical_data[product] = []
        
        # Iterate over all products
        for product in state.order_depths.keys():
            # Get the order depth for this product
            order_depth = state.order_depths[product]
            
            # Initialize the list of Orders to be sent
            orders: List[Order] = []
            
            # Get current position
            position = state.position.get(product, 0)
            position_limit = 50  # Same for both products
            
            # Calculate remaining position capacity
            buy_capacity = position_limit - position
            sell_capacity = position_limit + position
            
            
            if product == "RAINFOREST_RESIN":
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:

                    window = 20
                    buy_price = self.calculate_vwap(order_depth.buy_orders)
                    sell_price = self.calculate_vwap(order_depth.sell_orders)
                    mid_price = (buy_price + sell_price) / 2

                    prices = historical_data[product][-window:] if len(historical_data[product]) >= window else []
                    std = 1.48

                    if len(prices) >= window:
                        std = np.std(prices)

                    spread = std

                    inventory_factor = 0.05

                    if position > 0:
                        buy_price = int(mid_price - spread + inventory_factor * position)
                        sell_price = min(int(mid_price),int(mid_price + spread - inventory_factor * position))
                    elif position < 0:
                        buy_price = max(int(mid_price),int(mid_price - spread - inventory_factor * abs(position)))
                        sell_price = int(mid_price + spread + inventory_factor * abs(position))
                    else:
                        # Neutral inventory, no adjustment needed
                        buy_price = int(mid_price - spread)
                        sell_price = int(mid_price + spread)

                    buy_volume = buy_capacity
                    sell_volume = sell_capacity
                    
                    if buy_volume > 0:
                            orders.append(Order(product, buy_price, buy_volume))
                    if sell_volume > 0:
                            orders.append(Order(product, sell_price, -sell_volume))

                result[product] = orders
                historical_data[product].append(mid_price)
                logger.print(historical_data)
                
            if product == "KELP":
                continue
            
            if product == "SQUID_INK":                
                continue

        for product in historical_data.keys():
            if len(historical_data[product]) > 20:

                historical_data[product] = historical_data[product][-20:]

        state.traderData = jsonpickle.encode(historical_data)
        logger.flush(state, result, 0, state.traderData)

        return result, 0, state.traderData