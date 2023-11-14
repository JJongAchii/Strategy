import os
import sys
import logging
import numpy as np
import pandas as pd

from datetime import datetime, date, timedelta
from dateutil import parser
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))
from hive import db
from config import get_args, COREVIEW_ML_MODELS_FOLDER

logger = logging.getLogger("sqlite")
args = get_args()

####################################################################################################
# global variables
TODAY = parser.parse(args.date)
SEED_VALUES = 100
np.random.seed(SEED_VALUES)
####################################################################################################

# Make View by Utilizing MLP Algo
def run_mlp_prediction(today:date) -> None:
    """
    run this function when making the models learned to calculating increasing probabilities for CoreView indices

    This function inherits the class `PredSettings` from `mlpstrategy` module as its setting input parameters.
    It also inherits the class `BaggedMlpClassifier` from `mlp_prediction` module.
    And then, it generates features and models, stores models.
    Eventually, it loads the models to predict the future probability of increase of the securities.

    Args:
        today (date): standard rebalancing date
    """

    from core.strategy.mlpstrategy import PredSettings
    from core.model.ML.mlp_prediction import BaggedMlpClassifier
    tickers =[
        "MXWD Index","SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXHK Index","M1EF Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","MXRU Index",
        "BTSYTRUH Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index",
        "LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index",
        "SPGSCIP Index","CL1 Comdty","XAU Curncy","FNER Index"
        ]

    model_path = os.path.join(
        COREVIEW_ML_MODELS_FOLDER, today.strftime("%Y-%m-%d")
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    price_df = pd.DataFrame()
    for ticker in tickers:
        price_data = db.get_index_value(ticker)
        price_df = pd.concat([price_df, price_data], axis=1)


    pred_settings = PredSettings(
        train=True, save=True, model_path=model_path
    ).dict()
    prediction_model = BaggedMlpClassifier(**pred_settings)
    prediction_model.predict(price_df=price_df)


def run_mlp_allocation(today:date) -> None:
    """
    run this function when making the `AI_MLP_VIEW` for the next month to compare to the CORE_VIEW of KB Securities Research Center.
    Set the bounds of investment preference for asset classes(Equity, Treasury, Credit, Alternative) by ranking their probabilities of increase.
    On the one hand, if the probability of increase of an asset class is the largest, its bound of the investment preference is from 3 to 5.
    On the other hand, if it is the smallest, its bound of the investment preference is from 1 to 3.
    In the other cases, theirs are from 2 to 4.
    
    Return the investment preference of `each detailed asset class` in the set bound by utilizing the 95% confidence interval of its probability of increase.
    
    Args:
        today (date): standard rebalancing date
        
    Returns:
        pd.DataFrame: DataFrame which consists of the investment preference of each detailed asset class 
    """

    from core.strategy.mlpstrategy import MlpStrategy
    from scipy.stats import t

    model_path = os.path.join(
        COREVIEW_ML_MODELS_FOLDER, today.strftime("%Y-%m-%d")
    )
    tickers =[
        "MXWD Index","SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXHK Index","M1EF Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","MXRU Index",
        "BTSYTRUH Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index",
        "LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index",
        "SPGSCIP Index","CL1 Comdty","XAU Curncy","FNER Index",
        ]
    tickers_ac = ["MXWD Index","BTSYTRUH Index","LUACTRUU Index","SPGSCIP Index"]
                
    prices = pd.DataFrame()
    for ticker in tickers:
        price_data = db.get_index_value(ticker)
        prices = pd.concat([prices, price_data], axis=1)

    strategy = MlpStrategy.load(universe=pd.DataFrame(), prices=prices, level=5, model_path=model_path)
    prob_increase = strategy.view_prediction(today=today)
    
    df_mlp_views = pd.DataFrame()
    for ticker in tickers:
        prob = prob_increase[prob_increase.index == ticker]
        df_mlp_views = pd.concat([df_mlp_views, prob])
    df_mlp_views.columns = ["value"]

    df_mlp_stnd_views = pd.DataFrame()
    for ac in tickers_ac:
        prob = prob_increase[prob_increase.index == ac]
        df_mlp_stnd_views = pd.concat([df_mlp_stnd_views, prob])

    df_mlp_stnd_views.columns = ["value"]
    df_mlp_stnd_views.index = ["equity","treasury","credit","commodity"]
    df_mlp_stnd_views = df_mlp_stnd_views.sort_values(by="value",ascending=False).reset_index()
    df_mlp_stnd_views = df_mlp_stnd_views.drop(df_mlp_stnd_views[df_mlp_stnd_views["index"] == "commodity"].index)

    commodity_pref = [2,3,4]
    if df_mlp_stnd_views.iloc[0,0] == 'equity':
        equity_pref = [3,4,5]
        if df_mlp_stnd_views.iloc[2,0] == 'treasury':
            treasury_pref = [1,2,3]
            credit_pref = [2,3,4]
        elif df_mlp_stnd_views.iloc[2,0] == 'credit':
            credit_pref = [1,2,3]
            treasury_pref = [2,3,4]

    if df_mlp_stnd_views.iloc[0,0] == 'treasury':
        treasury_pref = [3,4,5]
        if df_mlp_stnd_views.iloc[2,0] == 'equity':
            equity_pref = [1,2,3]
            credit_pref = [2,3,4]
        elif df_mlp_stnd_views.iloc[2,0] == 'credit':
            credit_pref = [1,2,3]
            equity_pref = [2,3,4]

    if df_mlp_stnd_views.iloc[0,0] == 'credit':
        credit_pref = [3,4,5]
        if df_mlp_stnd_views.iloc[2,0] == 'equity':
            equity_pref = [1,2,3]
            treasury_pref = [2,3,4]
        elif df_mlp_stnd_views.iloc[2,0] == 'treasury':
            treasury_pref = [1,2,3]
            equity_pref = [2,3,4]

    from scipy.stats import t
    
    mean = df_mlp_views["value"].mean()
    std = df_mlp_views["value"].std()
    sample_size = len(df_mlp_views)
    confidence_level = 0.95
    standard_error = std / np.sqrt(sample_size)
    degree_freedom = sample_size - 1
    critical_value = t.ppf((1 + confidence_level) / 2, degree_freedom)
    margin_of_error = critical_value * standard_error
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    prefer_list = []
    for i in range(len(df_mlp_views)):
        asset_name = df_mlp_views.index[i]

        if asset_name in ["MXWD Index","SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXHK Index","KOSPI Index"]:
            if df_mlp_views['value'][i] < lower_bound:
                pref_mlp = equity_pref[0]
            if (df_mlp_views['value'][i] >= lower_bound) & (df_mlp_views['value'][i] < upper_bound):
                pref_mlp = equity_pref[1]
            if df_mlp_views['value'][i] >= upper_bound:
                pref_mlp = equity_pref[2]
                
        if asset_name in ["BTSYTRUH Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index"]:
            if df_mlp_views['value'][i] < lower_bound:
                pref_mlp = treasury_pref[0]
            if (df_mlp_views['value'][i] >= lower_bound) & (df_mlp_views['value'][i] < upper_bound):
                pref_mlp = treasury_pref[1]
            if df_mlp_views['value'][i] >= upper_bound:
                pref_mlp = treasury_pref[2]

        if asset_name in ["LUACTRUU Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index"]:
            if df_mlp_views['value'][i] < lower_bound:
                pref_mlp = credit_pref[0]
            if (df_mlp_views['value'][i] >= lower_bound) & (df_mlp_views['value'][i] < upper_bound):
                pref_mlp = credit_pref[1]
            if df_mlp_views['value'][i] >= upper_bound:
                pref_mlp = credit_pref[2]

        if asset_name in ["SPGSCIP Index","CL1 Comdty","XAU Curncy","FNER Index"]:
            if df_mlp_views['value'][i] < lower_bound:
                pref_mlp = commodity_pref[0]
            if (df_mlp_views['value'][i] >= lower_bound) & (df_mlp_views['value'][i] < upper_bound):
                pref_mlp = commodity_pref[1]
            if df_mlp_views['value'][i] >= upper_bound:
                pref_mlp = commodity_pref[2]

        prefer_list.append(pref_mlp)

    table_preference = pd.DataFrame(prefer_list, index=df_mlp_views.index, columns=["ai_mlp_view"])

    return table_preference

##########################################################################################################################################################################
# Make View by Utilizing ALPHA Algo
from core.factor.factor_lens import exposure, exposures_implied_performance, excess_performance
from pydantic import BaseModel

class RegionLensSettings(BaseModel):
    """
    assign bloomberg ticker to pre-defined factors.
    
    The class which maps the asset classes and the detailed ones to the index tickers to be used on the class `RegionLens` below
    
    Args:
        BaseModel (class): BaseModel to deciede what indices to consider
    """
    equity: str = "MXWD Index"
    equity_us: str = "SPX Index"  
    equity_eu: str = "SX5E Index"
    equity_jp: str = "NKY Index"
    equity_cn_ml: str = "SHCOMP Index" 
    equity_cn_hk: str = "MXHK Index"
    equity_em: str = "M1EF Index"
    equity_in: str = "MXIN Index"
    equity_vn: str = "MXVI Index"
    equity_br: str = "MXBR Index"
    equity_kr: str = "KOSPI Index"
    equity_ru: str = "MXRU Index"

    treasury:str = "BTSYTRUH Index"
    treasury_dm: str = "LUATTRUU Index"
    treasury_em: str = "EMUSTRUU Index"
    treasury_kr: str = "GVSK10YR Index"

    credit: str = "LUACTRUU Index"
    credit_us_ig: str = "LUACTRUU Index"
    credit_us_hy: str = "LF98TRUU Index"
    credit_kr: str = "KBDA3YR- Index"

    commodity: str = "SPGSCIP Index"
    commodity_co: str = "CL1 Comdty"
    commodity_gd: str = "XAU Curncy"
    commodity_rt: str = "FNER Index"


class RegionLens:
    """
    get index value from TB_MACRO_DATA table and macro id from TB_MACRO table.
    make a DataFrame which consists of the indices to residulize the each components.
    calculate the expected excessive return of each detailed asset class by residualizaing.

    Set the bound of the investment preference of each asset class by utilizing OECD Composite Leading economic Indicator(LEI) data.
    And return the investment preference by checking the 95% confidence interval of each detailed asset class in its asset class.
    
    Returns:
        pd.DataFrame: The DataFrame which consists of the investment preference of each detailed asset class
    """
       
    def __init__(self,today:date) -> None: 
        """
        Initialization

        Args:
            today (date): standard rebalancing date
        """
        factors = []
        price_data_df = pd.DataFrame()
        factor_lens = RegionLensSettings().dict()
        ticker_list = []
        for name, ticker in factor_lens.items():
            factors.append(name)
            ticker_list.append(ticker)
            price_data = db.get_index_value(ticker)
            price_data.columns = [name]

            price_data_df = pd.concat([price_data_df, price_data], axis=1)

        data = price_data_df.copy()
        factor_ticker = pd.concat([pd.DataFrame(factors, columns=["factor"]), pd.DataFrame(ticker_list, columns=["ticker"])], axis=1)

        factors_lst = []

        self.data = data
        self.today = today
        self.lastdate = datetime(today.year, today.month, 1) - timedelta(days=1)
        self.factor_ticker = factor_ticker
        self.factors_lst: Optional[pd.Series(dtype=float)] = []
        self.equity_us: Optional[pd.Series(dtype=float)] = None
        self.equity_eu: Optional[pd.Series(dtype=float)] = None
        self.equity_jp: Optional[pd.Series(dtype=float)] = None
        self.equity_cn_ml: Optional[pd.Series(dtype=float)] = None
        self.equity_cn_hk: Optional[pd.Series(dtype=float)] = None
        self.equity_em: Optional[pd.Series(dtype=float)] = None
        self.equity_in: Optional[pd.Series(dtype=float)] = None
        self.equity_vn: Optional[pd.Series(dtype=float)] = None
        self.equity_br: Optional[pd.Series(dtype=float)] = None
        self.equity_kr: Optional[pd.Series(dtype=float)] = None
        self.equity_ru: Optional[pd.Series(dtype=float)] = None
        self.treasury_dm: Optional[pd.Series(dtype=float)] = None
        self.treasury_em: Optional[pd.Series(dtype=float)] = None
        self.treasury_kr: Optional[pd.Series(dtype=float)] = None
        self.credit_us_ig: Optional[pd.Series(dtype=float)] = None
        self.credit_us_hy: Optional[pd.Series(dtype=float)] = None
        self.credit_kr: Optional[pd.Series(dtype=float)] = None
        self.commodity_co: Optional[pd.Series(dtype=float)] = None
        self.commodity_gd: Optional[pd.Series(dtype=float)] = None
        self.commodity_rt: Optional[pd.Series(dtype=float)] = None
        self.table_preference: Optional[pd.DataFrame(dtype=float)] = None
        self.process()

    def get_region_value(self, factor_val: pd.Series, factor_name: str) -> None:
        """
        assign factor value to factors_lst, a class attribute 
        
        Args:
            factor_val (pd.Series): calculated factor value 
            factor_name (str): factor name
        """
        factor = pd.DataFrame(factor_val.copy())
        factor.insert(0,'factor',[factor_name] * len(factor))
        factor = factor.reset_index()
        factor.columns = ['trd_dt','factor','value']
        self.factors_lst.append(factor)


    def calc_region(self) -> None:
        """
        calculate macro indices from the index dataframe.
        calculate 4 asset classes(Equity, Treasury, Credit, Alternative) and 20 detailed asset classes
        
        The Followings are the specifics assets:
        
            Equity: US, EU, JP, CN, HK, EM, IN, VN, BR, KR, RU
            Treasury: DM, EM, KR
            Credit: US_IG, US_HY, KR
            Alternative(commodity): Crude Oil, Gold, REITs)       
        """
        # Representative Asset classes
        self.equity = self.data.equity.dropna()
        self.get_region_value(self.equity,"equity")

        self.treasury = self.data.treasury.dropna()
        self.get_region_value(self.treasury,"treasury")

        self.credit = self.data.credit.dropna()
        self.get_region_value(self.credit,"credit")

        self.commodity = self.data.commodity.dropna()
        self.get_region_value(self.commodity,"commodity")


        # Equity
        self.equity_us = self.residualization(
            price=self.data.equity_us,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_us.name = "equity_us"
        self.get_region_value(self.equity_us,"equity_us")
        
        self.equity_eu = self.residualization(
            price=self.data.equity_eu,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_eu.name = "equity_eu"
        self.get_region_value(self.equity_eu,"equity_eu")

        self.equity_jp = self.residualization(
            price=self.data.equity_jp,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_jp.name = "equity_jp"
        self.get_region_value(self.equity_jp,"equity_jp")

        self.equity_cn_ml = self.residualization(
            price=self.data.equity_cn_ml,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_cn_ml.name = "equity_cn_ml"
        self.get_region_value(self.equity_cn_ml,"equity_cn_ml")

        self.equity_cn_hk = self.residualization(
            price=self.data.equity_cn_hk,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_cn_hk.name = "equity_cn_hk"
        self.get_region_value(self.equity_cn_hk,"equity_cn_hk")

        self.equity_em = self.residualization(
            price=self.data.equity_em,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_em.name = "equity_em"
        self.get_region_value(self.equity_em,"equity_em")

        self.equity_in = self.residualization(
            price=self.data.equity_in,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_in.name = "equity_in"
        self.get_region_value(self.equity_in,"equity_in")

        self.equity_vn = self.residualization(
            price=self.data.equity_vn,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_vn.name = "equity_vn"
        self.get_region_value(self.equity_vn,"equity_vn")

        self.equity_br = self.residualization(
            price=self.data.equity_br,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_br.name = "equity_br"
        self.get_region_value(self.equity_br,"equity_br")

        self.equity_kr = self.residualization(
            price=self.data.equity_kr,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_kr.name = "equity_kr"
        self.get_region_value(self.equity_kr,"equity_kr")

        self.equity_ru = self.residualization(
            price=self.data.equity_ru,
            price_factor=self.equity.to_frame().dropna(),
        )
        self.equity_ru.name = "equity_ru"
        self.get_region_value(self.equity_ru,"equity_ru")


        # Treasury
        self.treasury_dm = self.residualization(
            price=self.data.treasury_dm,
            price_factor=self.treasury.to_frame().dropna(),
        )
        self.treasury_dm.name = "treasury_dm"
        self.get_region_value(self.treasury_dm,"treasury_dm")

        self.treasury_em = self.residualization(
            price=self.data.treasury_em,
            price_factor=self.treasury.to_frame().dropna(),
        )
        self.treasury_em.name = "treasury_em"
        self.get_region_value(self.treasury_em,"treasury_em")

        self.treasury_kr = self.residualization(
            price=self.data.treasury_kr,
            price_factor=self.treasury.to_frame().dropna(),
        )
        self.treasury_kr.name = "treasury_kr"
        self.get_region_value(self.treasury_kr,"treasury_kr")      


        # Credit
        self.credit_us_ig = self.residualization(
            price=self.data.credit_us_ig,
            price_factor=self.credit.to_frame().dropna(),
        )
        self.credit_us_ig.name = "credit_us_ig"
        self.get_region_value(self.credit_us_ig,"credit_us_ig")      

        self.credit_us_hy = self.residualization(
            price=self.data.credit_us_hy,
            price_factor=self.credit.to_frame().dropna(),
        )
        self.credit_us_hy.name = "credit_us_hy"
        self.get_region_value(self.credit_us_hy,"credit_us_hy")      

        self.credit_kr = self.residualization(
            price=self.data.credit_kr,
            price_factor=self.credit.to_frame().dropna(),
        )
        self.credit_kr.name = "credit_kr"
        self.get_region_value(self.credit_kr,"credit_kr")      


        # Commodity
        self.commodity_co = self.residualization(
            price=self.data.commodity_co,
            price_factor=self.commodity.to_frame().dropna(),
        )        
        self.commodity_co.name = "commodity_co"
        self.get_region_value(self.commodity_co,"commodity_co")      

        self.commodity_gd = self.residualization(
            price=self.data.commodity_gd,
            price_factor=self.commodity.to_frame().dropna(),
        )
        self.commodity_gd.name = "commodity_gd"
        self.get_region_value(self.commodity_gd,"commodity_gd")      

        self.commodity_rt = self.residualization(
            price=self.data.commodity_rt,
            price_factor=self.commodity.to_frame().dropna(),
        )
        self.commodity_rt.name = "commodity_rt"
        self.get_region_value(self.commodity_rt,"commodity_rt")     

                 
            
    def process(self) -> None:
        """
        call calc_region faunction
        build a factor dataframe using a class attribute, factors_lst, and insert the calculated factor data to TB_Lens table.
        
        Returns:
            pd.DataFrame: The DataFrame which consists of the investment preference of each detailed asset class
        """        
        self.calc_region()
        
        df_factors = pd.DataFrame()
        df_factors = df_factors.append(self.factors_lst)
        df_factors = df_factors.sort_values(by=['trd_dt'],ascending=False)

        to_show = df_factors[df_factors.trd_dt == self.lastdate.strftime("%Y-%m-%d")].sort_index()
        while to_show.empty:
            self.lastdate -= timedelta(days=1)
            to_show = df_factors[df_factors.trd_dt == self.lastdate.strftime("%Y-%m-%d")].sort_index()
        
        for factor in ["equity","treasury","credit"]:
            to_show = to_show.drop(to_show[to_show["factor"] == factor].index)

        to_show = to_show.reset_index().drop("index", axis=1)

        from core.model.regime.lei import USLEIHP
        from scipy.stats import t

        regime = USLEIHP(asofdate=self.today).get_state(self.today.strftime("%Y-%m-%d"))

        commodity_pref = [2,3,4]
        if regime == 'expansion':
            equity_pref = [3,4,5]
            treasury_pref = [1,2,3]
            credit_pref = [2,3,4]
        if (regime == 'recovery')|(regime == 'slowdown'):
            equity_pref = [2,3,4]
            treasury_pref = [2,3,4]
            credit_pref = [2,3,4]
        if regime == 'contraction':
            equity_pref = [1,2,3]
            treasury_pref = [3,4,5] 
            credit_pref = [1,2,3]

        def get_stats_asset_class(data: pd.DataFrame):
            """
            return the lower and upper bounds of the 95% confidence interval of the asset class
            by expanding its detailed asset classes data through data resampling from normal distribution

            Args:
                data (pd.DataFrame): the data of an asset class

            Returns:
                float: the lower and upper bounds which are able to be compared to the actual value of each detailed asset class's
            """
            mean = data["value"].mean()
            std = data["value"].std()

            expanded_data=[]
            for _ in range(0, 300):
                expanded_data.append(np.random.normal(mean,std))
            expanded_data = pd.DataFrame(expanded_data, columns=["value"])

            expanded_mean = expanded_data["value"].mean()
            expanded_std = expanded_data["value"].std()
            sample_size = len(expanded_data)
            confidence_level = 0.95
            standard_error = expanded_std / np.sqrt(sample_size)
            degree_freedom = sample_size - 1
            critical_value = t.ppf((1 + confidence_level) / 2, degree_freedom)
            margin_of_error = critical_value * standard_error
            lower_bound = expanded_mean - margin_of_error
            upper_bound = expanded_mean + margin_of_error

            return lower_bound, upper_bound
        
        asset_list = []
        prefer_list = []
        for i in range(len(to_show)):
            asset_name = to_show['factor'][i].split('_')[0]
            asset_class_data = to_show[to_show['factor'].str.contains(asset_name)]
            ac_lower_bound, ac_upper_bound = get_stats_asset_class(asset_class_data)

            if asset_name == 'equity':
                if to_show['value'][i] < ac_lower_bound:
                    pref = equity_pref[0]
                if (to_show['value'][i] >= ac_lower_bound) & (to_show['value'][i] < ac_upper_bound):
                    pref = equity_pref[1]
                if to_show['value'][i] >= ac_upper_bound:
                    pref = equity_pref[2]
                    
            if asset_name == 'treasury':
                if to_show['value'][i] < ac_lower_bound:
                    pref = treasury_pref[0]
                if (to_show['value'][i] >= ac_lower_bound) & (to_show['value'][i] < ac_upper_bound):
                    pref = treasury_pref[1]
                if to_show['value'][i] >= ac_upper_bound:
                    pref = treasury_pref[2]

            if asset_name == 'credit':
                if to_show['value'][i] < ac_lower_bound:
                    pref = credit_pref[0]
                if (to_show['value'][i] >= ac_lower_bound) & (to_show['value'][i] < ac_upper_bound):
                    pref = credit_pref[1]
                if to_show['value'][i] >= ac_upper_bound:
                    pref = credit_pref[2]

            if asset_name == 'commodity':
                if to_show['value'][i] < ac_lower_bound:
                    pref = commodity_pref[0]
                if (to_show['value'][i] >= ac_lower_bound) & (to_show['value'][i] < ac_upper_bound):
                    pref = commodity_pref[1]
                if to_show['value'][i] >= ac_upper_bound:
                    pref = commodity_pref[2]

            prefer_list.append(pref)
            asset_list.append(to_show['factor'][i])

        self.table_preference = pd.DataFrame(prefer_list, index=asset_list, columns=["ai_alpha_view"]).reset_index()
        self.table_preference = pd.merge(left=self.table_preference, right=self.factor_ticker, how="inner",left_on="index",right_on="factor")[['ticker','ai_alpha_view']]
        self.table_preference.columns = ["index","ai_alpha_view"]

        return self.table_preference
    

    @staticmethod
    def residualization(
        price: pd.Series,
        price_factor: pd.DataFrame,
        window: int = 252 * 3,
        smoothing_window: int = 5,
        **kwargs,
    ) -> pd.Series:
        """
        residualize index values removed index cumulative return
        
        Args:
            price (pd.Series): the values of each specific(detailed) asset class
            price_factor (pd.DataFrame): the values of each asset class
            window (int, optional): the historical period to residualize the specific asset class. Defaults to 252*3 (days).
            smoothing_window (int, optional): the smoothing window to smooth the input `price` and `price_factor`. Defaults to 5 (days).

        Returns:
            pd.Series: excess_performance(excessive return of each detailed asset class)
        """

        itx = price.index.intersection(price_factor.index)

        price = price.loc[itx]
        price_factor = price_factor.loc[itx]

        pri_return_1 = price.pct_change().fillna(0)
        pri_return_2 = price_factor.pct_change().fillna(0)

        betas = exposure(
            dependents=pri_return_1, independents=pri_return_2, window=window, **kwargs
        )

        if smoothing_window is not None:
            betas = betas.rolling(window=smoothing_window, min_periods=0).mean()

        perf_exposure = exposures_implied_performance(
            exposures=betas, price_factor=price_factor
        )

        return excess_performance(price, perf_exposure)

##########################################################################################################################################################################
# Make View by Utilizing ABL Algo
def run_factor_allocation(today:date) -> None:
    """
    run abl allocation whenever calculating excess returns of the specific asset classes

    Set the bounds of investment preference for asset classes(Equity, Treasury, Credit, Alternative) by confirming today's market regime through `LEI` module.
    In case of `Expansion` , `Equity` will be from 3 to 5, `Treasury` from 1 to 3, and `Credit` from 2 to 4.
    In case of `Contraction` , in contrast of  `Expansion`, `Equity` will be from 1 to 3, and `Treasury` from 3 to 5, but `Credit` from 1 to 3.
    In case of `Slowdown` and `Recovery` , all the asset classes will be from 2 to 4.
    Constantly, `Alternative` will always be from 2 to 4.
    
    Return the investment preference of `each detailed asset class` in the set bound by utilizing the 95% confidence interval of its probability of increase.    

    Args:
        today (date): standard rebalancing date
        
    Returns:
        pd.DataFrame: The DataFrame which consists of the investment preference of each detailed asset class
    """
    from core.strategy.ablstrategy import AblStrategy
    from scipy.stats import t
    
    tickers=[
        "SPX Index","SX5E Index","NKY Index","SHCOMP Index","M1EF Index","MXHK Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","MXRU Index",
        "LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index",
        "LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index",
        "SPGSCIP Index","CL1 Comdty","XAU Curncy","FNER Index",
        ]
    equity_tickers = ["SPX Index","SX5E Index","NKY Index","SHCOMP Index","M1EF Index","MXHK Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","MXRU Index"]
    treasury_tickers = ["LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index"]
    credit_tickers = ["LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index"]
    commodity_tickers = ["SPGSCIP Index","CL1 Comdty","XAU Curncy","FNER Index"]
    
    universe = db.load_universe()
    price_asset = pd.DataFrame()
    for ticker in tickers:
        price_data = db.get_index_value(ticker)
        price_asset = pd.concat([price_asset, price_data], axis=1)

    lastdate = datetime(today.year, today.month, 1) - timedelta(days=1)
    price_factor = db.get_lens(today= lastdate)
    
    strategy = AblStrategy.load(
        universe=universe,
        price_asset=price_asset,
        price_factor=price_factor,
        regime='lei',
        asofdate=lastdate,
        level=5,
    )

    expected_returns = pd.DataFrame(strategy.view_prediction(today=today))
    expected_returns.columns = ["value"]
    df_factor_views = expected_returns.copy()

    current_regime = strategy.regime.get_state(lastdate.strftime("%Y-%m-%d"))

    commodity_pref = [2,3,4]
    if current_regime == 'expansion':
        equity_pref = [3,4,5]
        treasury_pref = [1,2,3]
        credit_pref = [2,3,4]
    if (current_regime == 'recovery')|(current_regime == 'slowdown'):
        equity_pref = [2,3,4]
        treasury_pref = [2,3,4]
        credit_pref = [2,3,4]
    if current_regime == 'contraction':
        equity_pref = [1,2,3]
        treasury_pref = [3,4,5] 
        credit_pref = [1,2,3]


    def get_stats_asset_class(data: pd.DataFrame):
        """
        return the lower and upper bounds of the 95% confidence interval of the asset class
        by expanding its detailed asset classes data through data resampling from normal distribution

        Args:
            data (pd.DataFrame): the data of an asset class

        Returns:
            float: the lower and upper bounds which are able to be compared to the actual value of each detailed asset class's
        """
        mean = data["value"].mean()
        std = data["value"].std()

        expanded_data=[]
        for _ in range(0, 300):
            expanded_data.append(np.random.normal(mean,std))
        expanded_data = pd.DataFrame(expanded_data, columns=["value"])

        expanded_mean = expanded_data["value"].mean()
        expanded_std = expanded_data["value"].std()
        sample_size = len(expanded_data)
        confidence_level = 0.95
        standard_error = expanded_std / np.sqrt(sample_size)
        degree_freedom = sample_size - 1
        critical_value = t.ppf((1 + confidence_level) / 2, degree_freedom)
        margin_of_error = critical_value * standard_error
        lower_bound = expanded_mean - margin_of_error
        upper_bound = expanded_mean + margin_of_error

        return lower_bound, upper_bound

    prefer_list = []
    equity_df = pd.DataFrame()
    treasury_df = pd.DataFrame()
    credit_df = pd.DataFrame()
    commodity_df = pd.DataFrame()

    for ticker in equity_tickers:
        equity_df = pd.concat([equity_df , df_factor_views[df_factor_views.index == ticker]])

    for ticker in treasury_tickers:
        treasury_df = pd.concat([treasury_df , df_factor_views[df_factor_views.index == ticker]])

    for ticker in credit_tickers:
        credit_df = pd.concat([credit_df , df_factor_views[df_factor_views.index == ticker]])

    for ticker in commodity_tickers:
        commodity_df = pd.concat([commodity_df , df_factor_views[df_factor_views.index == ticker]])


    for i in range(len(df_factor_views)):
        index_ticker = df_factor_views.index[i]

        if index_ticker in equity_tickers:
            ac_lower_bound, ac_upper_bound = get_stats_asset_class(equity_df)
            if df_factor_views['value'][i] < ac_lower_bound:
                pref_abl = equity_pref[0]
            if (df_factor_views['value'][i] >= ac_lower_bound) & (df_factor_views['value'][i] < ac_upper_bound):
                pref_abl = equity_pref[1]
            if df_factor_views['value'][i] >= ac_upper_bound:
                pref_abl = equity_pref[2]
                
        elif index_ticker in treasury_tickers:
            ac_lower_bound, ac_upper_bound = get_stats_asset_class(treasury_df)
            if df_factor_views['value'][i] < ac_lower_bound:
                pref_abl = treasury_pref[0]
            if (df_factor_views['value'][i] >= ac_lower_bound) & (df_factor_views['value'][i] < ac_upper_bound):
                pref_abl = treasury_pref[1]
            if df_factor_views['value'][i] >= ac_upper_bound:
                pref_abl = treasury_pref[2]

        elif index_ticker in credit_tickers:
            ac_lower_bound, ac_upper_bound = get_stats_asset_class(credit_df)
            if df_factor_views['value'][i] < ac_lower_bound:
                pref_abl = credit_pref[0]
            if (df_factor_views['value'][i] >= ac_lower_bound) & (df_factor_views['value'][i] < ac_upper_bound):
                pref_abl = credit_pref[1]
            if df_factor_views['value'][i] >= ac_upper_bound:
                pref_abl = credit_pref[2]

        elif index_ticker in commodity_tickers:
            ac_lower_bound, ac_upper_bound = get_stats_asset_class(commodity_df)
            if df_factor_views['value'][i] < ac_lower_bound:
                pref_abl = commodity_pref[0]
            if (df_factor_views['value'][i] >= ac_lower_bound) & (df_factor_views['value'][i] < ac_upper_bound):
                pref_abl = commodity_pref[1]
            if df_factor_views['value'][i] >= ac_upper_bound:
                pref_abl = commodity_pref[2]
        
        prefer_list.append(pref_abl)


    table_preference = pd.DataFrame(prefer_list, index = tickers, columns= ['ai_factor_view'])

    return table_preference

##########################################################################################################################################################################
def run_coreview_process(today: date):
    """
    Organize all the information needed to be inserted into DB for the dashboard `View Info` 

    Args:
        today (date): standard rebalancing date
    """
    extra = dict(user=args.user, activity="view generating", category="script")
    lastdate = datetime(today.year, today.month, 1) - timedelta(days=1)
    
    if today != db.get_start_trading_date(market="KR", asofdate=today):
        logger.info(msg=f"[SKIP] VIEW GENERATOR . {today:%Y-%m-%d}", extra=extra)
        return
    else:
        logger.info(msg=f"[PASS] start VIEW GENERATOR. {today:%Y-%m-%d}", extra=extra)
        
    # table basic info ########################################################################################################
    if today == date(2018,3,2):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","M1EF Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","SPGSCIP Index"],
        } 
    elif (today>date(2018,3,2))&(today<=date(2019,5,2)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","M1EF Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","SPGSCIP Index","FNER Index"],
        } 
    elif (today>date(2019,5,2))&(today<=date(2019,11,1)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","SPGSCIP Index","FNER Index"],
        } 
    elif (today>date(2019,11,1))&(today<=date(2020,2,3)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","MXRU Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","SPGSCIP Index","FNER Index"],
        } 
    elif (today>date(2020,2,3))&(today<=date(2021,9,1)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","MXRU Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","CL1 Comdty","XAU Curncy","FNER Index"],
        } 
    elif (today>date(2021,9,1))&(today<=date(2021,11,1)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","CL1 Comdty","XAU Curncy","FNER Index"],
        } 
    elif (today>date(2021,11,1))&(today<=date(2021,12,1)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXIN Index","MXVI Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","CL1 Comdty","XAU Curncy","FNER Index"],
        } 
    elif (today>date(2021,12,1))&(today<=date(2022,1,3)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index","CL1 Comdty","XAU Curncy","FNER Index"],
        } 
    elif (today>date(2022,1,3))&(today<=date(2023,3,2)):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","KBDA3YR- Index","CL1 Comdty","XAU Curncy","FNER Index"],
        } 
    elif today > date(2023,3,2):
        assets_for_tickers = {
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXHK Index","KOSPI Index","LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index","LUACTRUU Index","KBDA3YR- Index","CL1 Comdty","XAU Curncy","FNER Index"],
        }

    last_month_view_table = pd.DataFrame()
    if last_month_view_table.empty:
        assets_for_tickers_last_month_view = {
            'asset_class':["주식","주식","주식","주식","주식","주식","주식","주식","주식","주식","주식",
                           "채권","채권","채권",
                           "채권","채권","채권",
                           "대체자산","대체자산","대체자산","대체자산"],
            'category':["선진국","선진국","선진국","신흥국","신흥국","신흥국","신흥국","신흥국","신흥국","신흥국","신흥국",
                        "국채","국채","국채",
                        "크레딧","크레딧","크레딧",
                        "원자재","원자재","원자재","부동산"],
            'target':["미국","유럽","일본","중국(본토)","중국(홍콩)","신흥국","인디아","베트남","브라질","한국","러시아",
                      "선진국채","신흥국채","한국채",
                      "미국(IG)","미국(HY)","한국회사채",
                      "원자재","원유","금","부동산/리츠"],
            'index_ticker':["SPX Index","SX5E Index","NKY Index","SHCOMP Index","MXHK Index","M1EF Index","MXIN Index","MXVI Index","MXBR Index","KOSPI Index","MXRU Index",
                            "LUATTRUU Index","EMUSTRUU Index","GVSK10YR Index",
                            "LUACTRUU Index","LF98TRUU Index","KBDA3YR- Index",
                            "SPGSCIP Index","CL1 Comdty","XAU Curncy","FNER Index"],
            'index_name':["S&P 500 Index","Euro Stoxx 50 Pr","Nikkei 225 Index","SHANGHAI SE COMPOSITE","MSCI HongKong Index","MSCI Emerging Markets Net Tot","MSCI India Index","MSCI Vietnam Index","MSCI Brazil Index","KOSPI Index","MSCI Russia Index",
                          "U.S. Treasury Bond","Bloomberg EM USD Aggregate Tota","KCMP South Korea Treasury Bond",
                          "Bloomberg US Corporate Total R","Bloomberg US Corporate High Yi","Korea Corporate",
                          "S&P GSCI Excess Return CME Index","Generic 1st 'CL' Future","Gold Spot $/Oz","FTSE NAREIT All Equity REITs Index"],
            'macro_id':[46,50,47,48,35,13,36,37,38,49,39,53,14,51,4,6,52,40,54,55,41],
            'asset_ticker':["SPY","VGK","EWJ","MCHI","EWH","VWO","INDA","VNM","EWZ","EWY","ERUS","GOVT","EBND","148070","LQD","HYG","239660","GSG","DBO","GLD","VNQ"],
            'stk_id':[1245,1362,320,870,318,1416,707,1380,339,338,2217,539,239,2001,858,599,2167,546,174,528,1381],
            'default_weight':[0.35,0.08,0.03,0.06,0.03,0,0,0.01,0.01,0.02,0.01,0.15,0.05,0,0.075,0.025,0,0,0.04,0.03,0.03]         
        }
        assets_for_tickers_last_month_view = pd.DataFrame(assets_for_tickers_last_month_view)
        last_month_view_table = assets_for_tickers_last_month_view[
            ['asset_class','category','target','index_ticker','index_name','asset_ticker','default_weight']
        ]  
    assets_for_tickers = pd.DataFrame(assets_for_tickers)
    rebal_dt = db.get_start_trading_date(market="KR", asofdate=today)

    last_rebal_dt = datetime(lastdate.year, lastdate.month,1)
    assets_for_tickers["rebal_dt"] = rebal_dt
    assets_for_tickers = pd.merge(last_month_view_table, assets_for_tickers, "inner", on= "index_ticker")

    # stats #####################################################################################################################
    assets_prices = db.get_price(tickers=", ".join(assets_for_tickers.asset_ticker.tolist()))
    assets_prices = assets_prices[(assets_prices.index >= last_rebal_dt)&(assets_prices.index <= lastdate)][assets_for_tickers.asset_ticker]
    return_1m = pd.DataFrame((assets_prices.iloc[-1,:] - assets_prices.iloc[0,:])/ assets_prices.iloc[0,:], columns=["return_1m"])
    
    asset_volume = db.query.get_volume(tickers=", ".join(assets_for_tickers.asset_ticker.tolist()))
    asset_volume = asset_volume[(asset_volume.index >= last_rebal_dt.strftime("%Y-%m-%d"))&(asset_volume.index <= lastdate.strftime("%Y-%m-%d"))][assets_for_tickers.asset_ticker]
    avg_volume_1m = pd.DataFrame(asset_volume.mean(), columns=['avg_volume_1m'])
    
    asset_aum = db.query.get_aum(tickers=", ".join(assets_for_tickers.asset_ticker.tolist()))
    aum = pd.DataFrame(asset_aum.iloc[-1,:])
    aum.columns = ["aum"]
    
    stat_1m = pd.concat([return_1m,avg_volume_1m,aum], axis=1).reset_index()
    
    page = pd.merge(left= assets_for_tickers,right= stat_1m,how="inner",left_on="asset_ticker",right_on="ticker")[
        ["rebal_dt","asset_class","category","target","index_ticker","index_name","return_1m","avg_volume_1m","aum"]]
    
    outlook_core_view_this_month_df = pd.DataFrame()
    for sheet in ["preference", "default_weights"]:
        core_view_table = pd.read_excel("core/analytics/coreview/kb_coreview.xlsx",sheet_name=sheet, engine="openpyxl")
        core_view_table.Date = pd.to_datetime(core_view_table.Date)
        core_view_table = core_view_table.set_index("Date")
        core_view_table = core_view_table.loc[:today].iloc[-1:].T.dropna()
        outlook_core_view_this_month_df = pd.concat([outlook_core_view_this_month_df, core_view_table], axis=1)
        
    outlook_core_view_this_month_df = outlook_core_view_this_month_df.reset_index()
    outlook_core_view_this_month_df.columns = ["target", "core_view", "default_weight"]
    
    page = pd.merge(page, outlook_core_view_this_month_df, on="target")
    print("page:\n", page)
    print("-"*120)

    # Execution ##################################################################################################################
    total_table_preference = pd.DataFrame()
    run_mlp_prediction(today)
    table_preference_mlp = run_mlp_allocation(today).reset_index()
    total_table_preference = pd.concat([total_table_preference, table_preference_mlp], axis=1)

    table_preference_lens = RegionLens(today).table_preference
    total_table_preference = pd.merge(left=total_table_preference,right=table_preference_lens,how="inner",left_on="index",right_on="index")[["index","ai_mlp_view","ai_alpha_view"]]
    total_table_preference.columns = ["index_ticker","ai_mlp_view","ai_alpha_view"]

    table_preference_abl = run_factor_allocation(today).reset_index()
    total_table_preference = pd.concat([total_table_preference, table_preference_abl], axis=1)[["index_ticker","ai_mlp_view","ai_alpha_view","ai_factor_view"]]

    result = pd.merge(left=page, right=total_table_preference, how="inner", left_on="index_ticker", right_on="index_ticker")[
        ["rebal_dt","asset_class","category","target","index_ticker","index_name","default_weight","return_1m","avg_volume_1m","aum","core_view","ai_mlp_view","ai_alpha_view","ai_factor_view"]
    ]
    result.core_view = result.core_view.astype(int)
    result.ai_mlp_view = result.ai_mlp_view.astype(int)
    result.ai_alpha_view = result.ai_alpha_view.astype(int)
    result.ai_factor_view = result.ai_factor_view.astype(int)
    print("@ result @ \n", result)
    
    if args.database == "true":
        try:
            db.TbViewInfo.insert(result)
            logger.info(msg=f"[PASS] INSERT VIEW. {today:%Y-%m-%d}", extra=extra)
        except:
            db.TbViewInfo.update(result)
            logger.info(msg=f"[PASS] UPDATE VIEW. {today:%Y-%m-%d}", extra=extra)
        
    logger.info(msg=f"[PASS] End VIEW GENERATOR. {today:%Y-%m-%d}", extra=extra)
    
    return result


if __name__ == "__main__":
    
    for year in range(2018,2024):
        for month in range(1,13):
            first_day_of_month = date(year, month, 1)
            start_trading_date=db.get_start_trading_date(market="KR", asofdate=first_day_of_month)
            if start_trading_date >= date(2018,3,1) and start_trading_date < date.today():
                today = start_trading_date
                
                result = run_coreview_process(today)
                db.TbViewInfo.insert(result)