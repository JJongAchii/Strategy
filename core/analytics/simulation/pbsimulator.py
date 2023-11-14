import numpy as np
import pandas as pd
import yfinance as yf


def goal_probability(
        result_initial: float,
        result_monthly: float,
        initial_amount: int,
        monthly_savings: int,
        goal_amount: int,
        func: str = 'mean'
):
    portfolio_balances = initial_amount * result_initial + monthly_savings * result_monthly
    if func == 'mean':
        goal_prob = portfolio_balances / goal_amount
    elif func == 'percentile':
        goal_prob = np.sum(goal_amount <= portfolio_balances)

    return goal_prob


class Simulator:
    def __init__(
            self,
            price: pd.DataFrame,
            years: int = 100,
            iterations: int = 1000,
            method: str = 'bootstrap',
            ##risk level에 따라 weight 조절
    ):
        self.years = years
        self.iterations = iterations
        self.weights = [0.1, 0.2, 0.7]
        self.monthly_returns = self.simulate(price=price, method=method)

    def simulate(self, price: pd.DataFrame, method: str = 'bootstrap'):
        if method == 'bootstrap':
            pct_returns = price.pct_change()[1:]
            price_sample = pct_returns.sample(n=self.years*252*self.iterations, replace=True)
            price_3d = price_sample.values.reshape(self.years*12*self.iterations, 21, -1)
            monthly_returns = np.cumprod(1 + price_3d, axis=1)[:, -1]
            monthly_returns = np.reshape(monthly_returns, (self.years*12, self.iterations, -1))
        elif method == 'gbm':
            pct_returns = price.pct_change()[1:]
            c = len(pct_returns.columns)
            cov = np.cov(pct_returns.T) * 21
            corr_cov = np.linalg.cholesky(cov)
            mu = np.mean(pct_returns, axis=0) * 21
            total_iter = self.years * 12 * self.iterations
            z = np.random.normal(0, 1, size=(c, total_iter))

            drift = np.full((total_iter, c), mu).T
            shock = np.dot(corr_cov, z)

            monthly_returns = drift + shock
            monthly_returns = np.reshape(monthly_returns.T, (self.years * 12, self.iterations, -1))
            monthly_returns = np.exp(monthly_returns)

        return monthly_returns

    def calc_portfolio_balance(self, returns: np.ndarray, func: str = 'mean'):
        # portfolio_returns = np.prod(returns ** self.weights, axis=2)
        portfolio_returns = np.sum((returns - 1) * self.weights, axis=2)
        initial_balance = np.cumprod(1 + portfolio_returns, axis=0)
        monthly_balance = np.cumprod(1 + portfolio_returns, axis=0).cumsum(axis=0)
        if func == 'mean':
            initial = np.mean(initial_balance, axis=1)
            monthly = np.mean(monthly_balance, axis=1)
        elif func == 'percentile':
            initial = np.percentile(initial_balance, range(1, 100), axis=1)
            monthly = np.percentile(monthly_balance, range(1, 100), axis=1)
        elif func == 'all':
            initial = initial_balance
            monthly = monthly_balance

        return initial, monthly


if __name__ == "__main__":

    price = yf.download("AGG, SPY, GSG", start="2020-01-01")['Adj Close']

    period = int(input("투자기간을 입력하세요: "))
    initial_amount = int(input("투자금액을 입력하세요: "))
    monthly_savings = int(input("월납입금액을 입력하세요: "))
    goal_amount = int(input("목표금액을 입력하세요: "))

    sim = Simulator(price=price)
    initial, monthly = sim.calc_portfolio_balance(sim.monthly_returns)

    gp = goal_probability(
        result_initial=initial[period - 1],
        result_monthly=monthly[period - 1],
        initial_amount=initial_amount,
        monthly_savings=monthly_savings,
        goal_amount=goal_amount,
    )
    print(gp)


