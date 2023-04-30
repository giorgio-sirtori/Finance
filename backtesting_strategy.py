import backtrader as bt
import yfinance as yf
import datetime

# Create the strategy class
class MovingAverageCrossover(bt.Strategy):
    params = (
        ("short_window", 50),
        ("long_window", 200),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_window
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_window
        )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.sma_short[0] > self.sma_long[0]:
                self.log(f"BUY CREATE, {self.dataclose[0]:.2f}")
                self.order = self.buy()
        else:
            if self.sma_short[0] < self.sma_long[0]:
                self.log(f"SELL CREATE, {self.dataclose[0]:.2f}")
                self.order = self.sell()

# Download historical data
ticker = "KO"
start = "2019-01-01"
end = "2023-04-01"
data = yf.download(ticker, start=start, end=end)
data.to_csv(f"{ticker}.csv")

# Initialize the backtesting engine
cerebro = bt.Cerebro()

# Add the strategy
cerebro.addstrategy(MovingAverageCrossover)

# Load the data and add to the engine
data = bt.feeds.YahooFinanceCSVData(dataname=f"{ticker}.csv")
cerebro.adddata(data)

# Set the starting capital
cerebro.broker.setcash(10000.0)

# Set the commission
cerebro.broker.setcommission(commission=0.001)

# Print the starting capital
print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

# Run the backtest
cerebro.run()

# Print the final capital
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

# Plot the results
cerebro.plot()
