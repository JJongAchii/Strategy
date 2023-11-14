import numpy as np 
import pandas as pd 
from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import curve_fit



class GlidePath:

    def __init__(self, prices, period, ann_exp_return=0.05, years=40, iterations=1000):
        self.prices = prices                                # pandas dataframe of asset prices 
        self.period = period                                # string 'daily', 'weekly' ... 
        self.years = years                                  # GlidePath years
        self.iterations = iterations                        # Number of simulations
        self.ann_factor = 52                                # Set default annulisation factor to weekly

        self.yearly_deposit = 1                             # yearly deposit amount relative to the first deposit
        self.ann_exp_inflation = 0.02                       # expected deposit growth rate (inflation rate is used)
        self.ann_exp_return = ann_exp_return                # client's expected annual return level
        
        self.discount = 0.98                                # discount rate of client's utility attribution over time
        
        self.asset_names = list(self.prices.columns)        # store asset names <-- from prices column names
        self.asset_num = len(self.asset_names)              # stock number of assets
        
        self.returns = self._return_series()                # generate returns series shape = (years * iterations, asset_num)
        self.principal = self._principal_series()           # generate clients' principal amount shape (years + 1, 1)
        self.target = self._target_series()                 # generate clients' target amount shape (years + 1, 1)
        self.portfolio_optimized = None                     # store optimized portfolio simulations
        self.weight_optimized, self.weight_fitted = self.run()

        


    # generate returns series shape = (years * iterations, asset_num)
    def _return_series(self):
        # sample returns' minimum period should be weekly
        # therefore when daily prices are given, we combine the weekly return of each date
        # and for any period that is greater than the weekly, standard procedure is used
        if self.period == "daily":
            r_series = pd.DataFrame()
            prices = self.prices.interpolate()
            for i in range(0, 7):
                r_series = pd.concat([r_series, to_log_returns(self.prices[i::7])])
            r_series.sort_index(inplace=True)
        else:
            r_series = to_log_returns(self.prices)
        # Geometic Brownian Motion Simulation of Asset Returns
        # Returns a numpy array shape = (years * iterations, asset_num)
        # DOUBLE CHECKING REQUIRED (Code Not Used)
        '''mean_returns = r_series.mean() * self.ann_factor
        cov_matrix = r_series.cov() * self.ann_factor 
        L = np.linalg.cholesky(cov_matrix).transpose()
        z = np.random.normal(0, 1, size = (self.years * self.iterations, self.asset_num)) 
        out = np.full(shape = (self.years * self.iterations, self.asset_num), fill_value=mean_returns) + np.dot(z, L)
        out = np.reshape(out, newshape=(self.years, self.iterations, -1))'''
        r_series = r_series.dropna().to_numpy()
        cov = np.cov(np.transpose(r_series)) * self.ann_factor # weekly or monthly -> yearly
        L = np.linalg.cholesky(cov) # cholesky decomposition
        mu = np.mean(r_series, axis=0) * self.ann_factor # weekly or monthly -> yearly 
        z = np.random.normal(0, 1, size=(self.asset_num, self.years * self.iterations))
        MU = np.full((self.years * self.iterations, self.asset_num), mu) #평균 값들의 배열
        MU = np.transpose(MU) # 지수가 행으로
        x= MU + np.dot(L,z)
        x = np.reshape(np.transpose(x),(self.years,self.iterations,-1))
        return np.exp(x)

    # generate clients' principal path shape=(years + 1, 1)
    def _principal_series(self):
        out = np.zeros(self.years + 1)
        for year in range(0, self.years):
            out[year + 1] = out[year] + self.yearly_deposit * (1 + self.ann_exp_inflation) ** year
        return out

    # generate clients' target path shape=(years + 1, 1)
    def _target_series(self):
        out = np.zeros(self.years + 1)
        for year in range(0, self.years):
            deposit = self.yearly_deposit * (1 + self.ann_exp_inflation) ** year
            gains = (1 + self.ann_exp_return)
            out[year + 1] = (out[year] + deposit) * gains
        return out

    # generate portfolio's value path based on the asset weightings
    def _portfolio_series(self, weight):

        out = np.zeros(shape = (self.years + 1, self.iterations))
        # Similar to the _principal_series & target_series
        # In portfolio_series we replaced scalar return to gbm generated returns * weight
        # weight is tested split into two parts weight of (asset_num - 1) number of assets
        # the last asset's weight is just 1 - accumulated weight of (asset_num - 1) assets

        # reason for doing so, shorter array list to optimize

        # weight parameter has a length of (asset_num - 1) * years --> genetic algorithm does not take multi dim 
        for year in range(0, self.years):
            portfolio_return = np.ones(self.iterations)
            weight_stacked = 0
            for asset in range(0, self.asset_num - 1):
                asset_weight = weight[year * (self.asset_num - 1) + asset]
                portfolio_return = portfolio_return * (self.returns[year, :, asset] ** asset_weight)
                weight_stacked += asset_weight
            portfolio_return = portfolio_return * self.returns[year, :, self.asset_num - 1] ** (1 - weight_stacked)
            deposits = self.yearly_deposit * (1 + self.ann_exp_inflation) ** year
            out[year + 1] = (out[year] + deposits) * portfolio_return
        return out
    

    # Composite Performance Index
    def V(self, weight):
        # Sums the utility function's value over time.
        portfolio = self._portfolio_series(weight)
        A = np.mean(portfolio, axis=1)
        T = self.target
        Vt = np.zeros(self.years + 1)
        for year in range(1, self.years + 1):
            Vt[year] = Vt[year - 1] + (self.discount ** year) * get_utility(A = A[year], T = T[year], year = year)
        
        # since the genetic algorithm does not take constraints like sum of all weight less then 1
        # thus manually added weight overflow penalty -> to eliminate solutions with sum of weights > 1 
        penalty = 0
        weight_overflow_penalty = np.zeros(self.years)
        for year in range(0, self.years):
            for asset in range(0, self.asset_num - 1):
                weight_overflow_penalty[year] += weight[year * (self.asset_num - 1) + asset]
            penalty += np.where(weight_overflow_penalty[year] > 1, 1000, 0)
        return -Vt[self.years] + penalty


    # Start Optimization
    def run(self):
        dim = self.years * (self.asset_num - 1)
        model = ga(
            function= self.V,
            dimension= dim,
            variable_type = 'real',
            variable_boundaries = np.array([[0,1]] * dim)
        )
        model.run()

        # store optimized weight
        weight_list = model.output_dict['variable']
        self.portfolio_optimized = pd.DataFrame(self._portfolio_series(weight_list))
        # reshape optimized weight into shape (years, asset_weight)
        weight_list = np.reshape(weight_list, (self.years, -1))
        weight_last = np.empty(shape=(self.years, 1))
        for year in range(0, self.years):
            weight_last[year, 0] = 1 - np.sum(weight_list[year, :])  
        weight_optimized = np.append(weight_list, weight_last, axis=1)
        self.weight_optimized = weight_optimized
        # store optimized weight into a dataframe
        opt_weight = pd.DataFrame(data = weight_optimized, columns = self.asset_names, index=range(1, self.years + 1))

        # fit the weight data of each asset with curve_fit and sigmoid function
        fit_weight = pd.DataFrame(index = np.linspace(0, self.years, self.years * 12))
        for asset in self.asset_names:
            ppot, pcov = curve_fit(
                sigmoid, opt_weight.index, opt_weight[asset], 
                bounds = [[self.years / 3, -0.2, 0, 0], [self.years, 0.2, 1, 1]])
            fit_weight[asset] = sigmoid(fit_weight.index, *ppot)

        return opt_weight, fit_weight


# helper function
def to_log_returns(prices):
    """
    Calculates the log returns of a price series.
    Formula is: ln(p1/p0)
    Args:
        * prices: Expects a price series
    """
    return np.log(prices / prices.shift(1))


# loss aversion utility function
def get_utility(A, T, year, lambda0 = 4.5, delta = 10, nu1 = 0.44, nu2 = 0.88):
    if A-T > 0 :
        return ((A-T)**nu1)/nu1
    elif A-T < 0 :
        return  -(lambda0 + delta*year )*((T-A)**nu2)/nu2
    else :
        return 0

# curve_fit function
def sigmoid(x, x0, k, L, L0):
     #y = 1 / (1 + np.exp(k*(x-x0)))
     y = L / (1 + np.exp(k * (x - x0))) + L0
     return y
