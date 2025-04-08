from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import jsonpickle

class Trader:
    def __init__(self):
        # Set fair value for Rainforest Resin
        self.rainforest_fair_value = 10000  # Starting estimate

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
            historical_data = {"RAINFOREST_RESIN": [], "KELP": []}
        else:
            historical_data = jsonpickle.decode(state.traderData)
        
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
            
            if product == "RAINFOREST_RESIN":
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    best_sell = min(order_depth.sell_orders.keys())
                    best_buy = max(order_depth.buy_orders.keys())
                    mid_price = 10000
                    
                    last_window = historical_data[product][-window:] if len(historical_data[product]) >= window else []
                    
                    std = np.std(last_window) if len(last_window) >= window else 1.48
                                    
                    buy_price = int(mid_price - std)
                    sell_price = int(mid_price + std)
                    
                    buy_volume = min(10, buy_capacity)
                    sell_volume = min(10, sell_capacity)
                    
                    print(std, buy_volume, sell_volume)
                    
                    if buy_volume > 0:
                        orders.append(Order(product, buy_price, buy_volume))
                    
                    if sell_volume > 0:
                        orders.append(Order(product, sell_price, -sell_volume))

                    if best_buy > best_sell:
                        orders.append(Order(product, best_buy, -1))
                        orders.append(Order(product, best_sell, 1))
                
                result[product] = orders
                historical_data[product].append((best_sell + best_buy)/2)
                print(historical_data)
                
            if product == "KELP":
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    best_sell = min(order_depth.sell_orders.keys())
                    best_buy = max(order_depth.buy_orders.keys())
                    mid_price = (best_sell + best_buy) / 2
                    
                    last_window = historical_data[product][-window:] if len(historical_data[product]) >= window else []
                    
                    std = np.std(last_window) if len(last_window) >= window else 1
                    
                    buy_price = int(mid_price - std)
                    sell_price = int(mid_price + std)
                    
                    buy_volume = min(5, buy_capacity)
                    sell_volume = min(5, sell_capacity)
                    
                    if buy_volume > 0:
                        orders.append(Order(product, buy_price, buy_volume))
                    
                    if sell_volume > 0:
                        orders.append(Order(product, sell_price, -sell_volume))

                    if best_buy > best_sell:
                        orders.append(Order(product, best_buy, -1))
                        orders.append(Order(product, best_sell, 1))
                
                result[product] = orders
                historical_data[product].append((best_sell + best_buy)/2)
                print(historical_data)
            
        # Update traderData with the serialized historical data
        state.traderData = jsonpickle.encode(historical_data)
        return result, 0, state.traderData