import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AssetResult:
    def __init__(self, df, risk_free_rate=0.05, name="Asset"):
        '''
        - df: Dataframe. Should contain a 'Close' column (already adjusted). 
          'Close' can be the price of a single asset or the total value of all positions in a portfolio.
        - risk_free_rate: Annualized risk-free rate
        '''

        if 'Close' not in df.columns:
            raise ValueError("'df' should contain a 'Close' column.")

        self.name = name
        self.df = df  # self.df = df.copy() # does not change 'df'
        self.prices = self.df['Close']
        self.risk_free_rate = risk_free_rate
        self.calculate()


    def calculate(self):
        self.sma_50 = self.prices.rolling(window=50).mean()
        self.sma_200 = self.prices.rolling(window=200).mean()

        self.daily_return = self.prices.pct_change()
        self.log_return = np.log(self.prices / self.prices.shift(1))
        self.vol_20 = self.daily_return.rolling(window=20).std()

        self.cum_returns = (self.daily_return + 1).cumprod()
        cum_max = self.cum_returns.cummax()
        self.drawdown = (cum_max - self.cum_returns) / cum_max
        self.max_drawdown = self.drawdown.max()

        self.annual_return = self.daily_return.mean() * 252
        self.annual_vol = self.daily_return.std() * np.sqrt(252)
        self.down_vol = self.daily_return[self.daily_return < 0].std() * np.sqrt(252)

        self.sharpe_ratio = (self.annual_return - self.risk_free_rate) / self.annual_vol
        self.sortino_ratio = (self.annual_return - self.risk_free_rate) / self.down_vol
        

    def get_performance(self):
        print(f"Performance Summary    {self.name}")
        print(f"Annualized Return:     {self.annual_return:.4f}")
        print(f"Annualized Volatility: {self.annual_vol:.4f}")
        print(f"Down Volatility:       {self.down_vol:.4f}")
        print(f"Sharpe Ratio:          {self.sharpe_ratio:.4f}")
        print(f"Sortino Ratio:         {self.sortino_ratio:.4f}")
        print(f"Max Drawdown:          {self.max_drawdown:.4f}")

        return {
            "asset": self.name,
            "annualized_return": self.annual_return,
            "annualized_volatility": self.annual_vol,
            "down_volatility": self.down_vol,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
        }


    def get_resample(self, df, freq='W'):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index is not Datetime.")
          
        # df.index = pd.to_datetime(df.index)
        # df['timestamp'] = pd.to_datetime(df['timestamp']) # timestamp or Date
        # df.set_index('timestamp', inplace=True)

        return df['Close'].resample(freq).last()


    def plot_basic(self):
        fig, ax = plt.subplots(2, 1, figsize=(10,6))

        ax[0].plot(self.prices, label=self.name)
        ax[0].plot(self.sma_50, label='SMA 50')
        ax[0].plot(self.sma_200, label='SMA 200')
        ax[0].set_title(f'{self.name}')
        ax[0].legend()

        ax[1].plot(self.drawdown, label='Drawdown', color='red')
        ax[1].set_title('Drawdown')
        ax[1].legend()

        plt.tight_layout()
        plt.show()
