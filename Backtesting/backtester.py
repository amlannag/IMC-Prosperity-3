import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datamodel import *
from backtest_trader import Trader

class Backtester:
    def __init__(self, csv_file: str, trader, position_limit: int = 50):
        self.data = pd.read_csv(csv_file, delimiter=';')
        self.data.sort_values(by=['day', 'timestamp'], inplace=True)

        self.trader = trader
        self.position_limit = position_limit

        self.cash = 0.0
        self.positions: Dict[Product, int] = {}

        self.traderData = ""
        self.pnl_history = []
        self.time_history = []
        self.trade_log = []  # List of Trade objects

        # Dictionary to track asset prices over time.
        # Keys are product symbols and values are lists of tuples (composite_time, price)
        self.asset_prices: Dict[Product, List[Tuple[int, float]]] = {}

    def run(self):
        # Group the data by (day, timestamp)
        grouped = self.data.groupby(['day', 'timestamp'])
        
        # Trade dictionaries that persist across groups.
        own_trades = {}
        market_trades = {}

        for (day, timestamp), group in grouped:
            # Create a composite timestamp (for example, day * 10000 + timestamp)
            composite_time = day * 10000 + timestamp

            listings = {}
            order_depths = {}
            
            # Process each row (each product snapshot) in the group.
            for _, row in group.iterrows():
                product = row['product']

                # Build the order book from CSV bid/ask columns.
                buy_orders = {}
                sell_orders = {}
                for i in range(1, 4):
                    bid_price_col = f'bid_price_{i}'
                    bid_volume_col = f'bid_volume_{i}'
                    if pd.notna(row[bid_price_col]) and pd.notna(row[bid_volume_col]):
                        price = float(row[bid_price_col])
                        volume = int(row[bid_volume_col])
                        if volume != 0:
                            buy_orders[price] = volume

                for i in range(1, 4):
                    ask_price_col = f'ask_price_{i}'
                    ask_volume_col = f'ask_volume_{i}'
                    if pd.notna(row[ask_price_col]) and pd.notna(row[ask_volume_col]):
                        price = float(row[ask_price_col])
                        volume = int(row[ask_volume_col])
                        if volume != 0:
                            # Store sell volumes as negatives.
                            sell_orders[price] = -volume

                order_depths[product] = OrderDepth(buy_orders, sell_orders)
                listings[product] = Listing(symbol=product, product=product, denomination="SEASHELLS")
                
                if product not in own_trades:
                    own_trades[product] = []
                if product not in market_trades:
                    market_trades[product] = []
                if product not in self.positions:
                    self.positions[product] = 0

                # Record a representative asset price for this snapshot.
                price = None
                if buy_orders and sell_orders:
                    price = (max(buy_orders.keys()) + min(sell_orders.keys())) / 2
                elif buy_orders:
                    price = max(buy_orders.keys())
                elif sell_orders:
                    price = min(sell_orders.keys())
                if price is not None:
                    if product not in self.asset_prices:
                        self.asset_prices[product] = []
                    self.asset_prices[product].append((composite_time, price))

            obs = Observation(plainValueObservations={}, conversionObservations={})

            state = TradingState(
                traderData=self.traderData,
                timestamp=timestamp,
                listings=listings,
                order_depths=order_depths,
                own_trades=own_trades,
                market_trades=market_trades,
                position=self.positions.copy(),
                observations=obs
            )

            result, _, self.traderData = self.trader.run(state)
            print(result)
            for product, orders in result.items():
                if product not in order_depths:
                    continue

                depth = order_depths[product]
                current_pos = self.positions.get(product, 0)

                for order in orders:
                    qty = order.quantity
                    symbol = order.symbol

                    # Determine execution price based on order side.
                    exec_price = None
                    if qty > 0:
                        # Buy order: execute at the best ask (lowest ask).
                        if depth.sell_orders:
                            exec_price = min(depth.sell_orders.keys())
                    elif qty < 0:
                        # Sell order: execute at the best bid (highest bid).
                        if depth.buy_orders:
                            exec_price = max(depth.buy_orders.keys())
                    
                    if exec_price is None:
                        continue

                    # Enforce position limits.
                    if qty > 0:
                        allowed = min(qty, self.position_limit - current_pos)
                    elif qty < 0:
                        allowed = max(qty, -self.position_limit - current_pos)
                    else:
                        allowed = 0

                    if allowed == 0:
                        continue

                    # Update cash and positions based solely on executed cash flows.
                    self.cash -= allowed * exec_price
                    self.positions[product] = current_pos + allowed
                    current_pos = self.positions[product]

                    self.trade_log.append(Trade(
                        symbol=symbol,
                        price=int(exec_price),
                        quantity=allowed,
                        buyer="TRADER" if allowed > 0 else "MARKET",
                        seller="MARKET" if allowed > 0 else "TRADER",
                        timestamp=composite_time
                    ))

            # PnL is now solely the cash balance.
            pnl = self.cash
            self.time_history.append(composite_time)
            self.pnl_history.append(pnl)

    def plot_all(self):
        """
        Plot all graphs on one page:
          - Upper-left: PnL over time.
          - The other three subplots: price time-series for up to three assets, with buys (green) and sells (red).
        """
        # Create a 2x2 grid of subplots.
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Subplot 1: PnL over time (upper-left)
        ax_pnl = axes[0, 0]
        ax_pnl.plot(self.time_history, self.pnl_history, label="PnL", color="blue")
        ax_pnl.set_xlabel("Composite Time (day x 10000 + timestamp)")
        ax_pnl.set_ylabel("Cash-based PnL")
        ax_pnl.set_title("PnL Over Time")
        ax_pnl.legend()
        ax_pnl.grid(True)
        
        # For the asset price plots we will use the other three subplots.
        # Collect asset keys; limit to three if more than three exist.
        asset_keys = list(self.asset_prices.keys())[:3]
        asset_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
        
        for ax, product in zip(asset_axes, asset_keys):
            price_history = self.asset_prices[product]
            times, prices = zip(*price_history)
            ax.plot(times, prices, label=f"{product} Price", color="blue")
            ax.set_xlabel("Composite Time")
            ax.set_ylabel("Price")
            ax.set_title(f"{product} Price Over Time")
            ax.grid(True)
            
            # Overlay the trades.
            buy_times, buy_prices = [], []
            sell_times, sell_prices = [], []
            for trade in self.trade_log:
                if trade.symbol == product:
                    if trade.quantity > 0:
                        buy_times.append(trade.timestamp)
                        buy_prices.append(trade.price)
                    elif trade.quantity < 0:
                        sell_times.append(trade.timestamp)
                        sell_prices.append(trade.price)
            if buy_times:
                ax.scatter(buy_times, buy_prices, color="green", marker="^", s=100, label="Buy")
            if sell_times:
                ax.scatter(sell_times, sell_prices, color="red", marker="v", s=100, label="Sell")
            ax.legend()
        
        # If there are fewer than 3 assets, hide extra subplots.
        if len(asset_keys) < 3:
            for ax in asset_axes[len(asset_keys):]:
                ax.axis('off')
        
        plt.suptitle("Backtesting Results", fontsize=16)
        plt.show()


if __name__ == "__main__":
    csv_file_path = "Backtesting/prep_data.csv"
    trader = Trader()
    backtester = Backtester(csv_file=csv_file_path, trader=trader, position_limit=50)
    backtester.run()
    backtester.plot_all()
