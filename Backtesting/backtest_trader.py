from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import jsonpickle

class Trader:
    def __init__(self):
        # Set fair value for Rainforest Resin
        self.rainforest_fair_value = 10000  
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
            
            window = 20

            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                #calc vvap mid_price
                buy_price = self.calculate_vwap(order_depth.buy_orders)
                sell_price = self.calculate_vwap(order_depth.sell_orders)
                mid_price = (buy_price + sell_price) / 2

                # Get historical prices
                prices = historical_data[product][-window:] if len(historical_data[product]) >= window else []

                
                std = 1.48 if product == "RAINFOREST_RESIN" else 1.0
                z_score = 0.0

                if len(prices) >= window:
                    std = np.std(prices)
                    mean = np.mean(prices)

                    z_score = ((mid_price - mean)) / std


            
            if product == "RAINFOREST_RESIN":
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    base_spread = 1

                    spread = std * (base_spread + z_score) if len(prices) >= window else 0

                    buy_price = int(mid_price - spread)
                    sell_price = int(mid_price + spread)

                    buy_volume = min(10, buy_capacity)
                    sell_volume = min(10, sell_capacity)
                    
                    if z_score > self.z_score_threshold:
                        if sell_volume > 0:
                            sell_volume = sell_capacity
                            orders.append(Order(product, sell_price, -sell_volume))
                    elif z_score < -self.z_score_threshold:
                        if buy_volume > 0:
                            buy_volume = buy_capacity
                            orders.append(Order(product, buy_price, buy_volume))
                    else:
                            if buy_volume > 0:
                                orders.append(Order(product, buy_price, buy_volume))
                            if sell_volume > 0:
                                orders.append(Order(product, sell_price, -sell_volume))


                
                result[product] = orders
                historical_data[product].append(mid_price)
                
            
        # Update traderData with the serialized historical data
        state.traderData = jsonpickle.encode(historical_data)
        return result, 0, state.traderData