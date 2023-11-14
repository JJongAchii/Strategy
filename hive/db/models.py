"""
database models
"""
import logging
import sqlalchemy as sa
from .mixins import Base, StaticBase, TimeSeriesBase, Mixins
from .client import engine

logger = logging.getLogger("sqlite")


def create_all() -> None:
    """drop all tables"""
    Base.metadata.create_all(bind=engine)


def drop_all() -> None:
    """drop all tables"""
    Base.metadata.drop_all(bind=engine)


class TbMeta(StaticBase):
    """metadata table"""

    __tablename__ = "tb_meta"
    stk_id = sa.Column(sa.Integer, sa.Identity(start=1), primary_key=True)
    ticker = sa.Column(sa.String(255), nullable=False)
    isin = sa.Column(sa.String(255), nullable=True)
    name = sa.Column(sa.String(1000), nullable=False)
    iso_code = sa.Column(sa.String(255), nullable=False)
    source = sa.Column(sa.String(255), nullable=True)
    fee = sa.Column(sa.Float)
    status = sa.Column(sa.String(100))
    remark = sa.Column(sa.Text, nullable=True)


class TbRiskScore(StaticBase):
    """asset risk score"""

    __tablename__ = "tb_risk_score"
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    trd_dt = sa.Column(sa.Date, primary_key=True)
    risk_score = sa.Column(sa.Integer, nullable=False)
    fee = sa.Column(sa.Numeric(9, 4), nullable=True)


class TbStrategy(StaticBase):
    """strategy table"""

    __tablename__ = "tb_strategy"
    strategy_id = sa.Column(sa.Integer, sa.Identity(start=1), primary_key=True)
    strategy = sa.Column(sa.String(255), nullable=False)
    strategy_name = sa.Column(sa.String(200), nullable=False)
    frequency = sa.Column(sa.String(255), nullable=False, default="M")
    min_assets = sa.Column(sa.Integer, nullable=False, default=2)
    min_periods = sa.Column(sa.Integer, nullable=False, default=2)


class TbPort(StaticBase):
    """portfolio table (portfolio is actual portfolio construct based on strategy)"""

    __tablename__ = "tb_port"
    port_id = sa.Column(sa.Integer, sa.Identity(start=1), primary_key=True)
    strategy_id = sa.Column(sa.ForeignKey("tb_strategy.strategy_id"), nullable=False)
    port_cd_tb = sa.Column(sa.Integer, nullable=True)
    gbi = sa.Column(sa.String(10), nullable=True)
    gbi_cd = sa.Column(sa.Integer, nullable=True)
    gbi2 = sa.Column(sa.String(10), nullable=True)
    gbi2_cd = sa.Column(sa.Integer, nullable=True)
    portfolio = sa.Column(sa.String(255), nullable=False)
    currency = sa.Column(sa.String(255), nullable=False, default="KRW")
    name = sa.Column(sa.String(200), nullable=True)
    remark = sa.Column(sa.Text, nullable=True)


class TbPortValue(TimeSeriesBase):
    """portfolio value"""

    __tablename__ = "tb_port_value"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    port_id = sa.Column(sa.ForeignKey("tb_port.port_id"), primary_key=True)
    value = sa.Column(sa.Float, nullable=False)
    mdd_1y = sa.Column(sa.Float, nullable=True)
    sharp_1y = sa.Column(sa.Float, nullable=True)
    mdd = sa.Column(sa.Float, nullable=True)
    sharp = sa.Column(sa.Float, nullable=True)


class TbPortBook(TimeSeriesBase):
    """strategy book"""

    __tablename__ = "tb_port_book"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    port_id = sa.Column(sa.ForeignKey("tb_port.port_id"), primary_key=True)
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    weights = sa.Column(sa.Float, nullable=True)
    trade_weights = sa.Column(sa.Float, nullable=True)


class TbPortApValue(TimeSeriesBase):
    """portfolio ap value"""

    __tablename__ = "tb_port_ap_value"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    port_id = sa.Column(sa.ForeignKey("tb_port.port_id"), primary_key=True)
    value = sa.Column(sa.Float, nullable=False)


class TbPortApBook(TimeSeriesBase):
    """portfolio ap book"""

    __tablename__ = "tb_port_ap_book"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    port_id = sa.Column(sa.ForeignKey("tb_port.port_id"), primary_key=True)
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    weights = sa.Column(sa.Float, nullable=True)


class TbProduct(StaticBase):
    __tablename__ = "tb_product"
    wrap_cd = sa.Column(sa.CHAR(5), primary_key=True)
    ptfl_cd = sa.Column(sa.CHAR(5), primary_key=True)
    port_id = sa.Column(sa.ForeignKey("tb_port.port_id"), primary_key=True)
    dws_gl_ern_ccd = sa.Column(sa.CHAR(2), nullable=True)
    dws_invst_stl_ccd = sa.Column(sa.CHAR(2), nullable=True)
    dws_modl_ptfl_apnt_ccd = sa.Column(sa.CHAR(2), nullable=True)
    remark = sa.Column(sa.Text, nullable=True)


class TbPortAlloc(StaticBase):
    """allocation weight"""

    __tablename__ = "tb_port_alloc"
    rebal_dt = sa.Column(sa.Date, primary_key=True)
    port_id = sa.Column(sa.ForeignKey("tb_port.port_id"), primary_key=True)
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    weights = sa.Column(sa.Numeric(9, 4), nullable=False)
    shares = sa.Column(sa.Integer, nullable=False, default=0)
    ap_weights = sa.Column(sa.Numeric(9, 4), nullable=False, default=0)


class TbUniverse(StaticBase):
    """investment universe"""

    __tablename__ = "tb_universe"
    strategy_id = sa.Column(sa.ForeignKey("tb_strategy.strategy_id"), primary_key=True)
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    active = sa.Column(sa.Integer, nullable=True, default=1)
    strg_asset_class = sa.Column(sa.String(20), nullable=True)
    wrap_asset_class_code = sa.Column(sa.CHAR(2), nullable=True)
    remark = sa.Column(sa.Text, nullable=True)


class TbDailyBar(TimeSeriesBase):
    """combined daily series"""

    __tablename__ = "tb_daily_bar"
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    trd_dt = sa.Column(sa.Date, primary_key=True)
    close_prc = sa.Column(sa.Float, nullable=True)
    gross_rtn = sa.Column(sa.Float, nullable=True)
    adj_value = sa.Column(sa.Float, nullable=True)


class TbMacro(StaticBase):
    """macro"""

    __tablename__ = "tb_macro"
    macro_id = sa.Column(sa.Integer, sa.Identity(start=1), primary_key=True)
    name = sa.Column(sa.String(255), nullable=True)
    future_ticker = sa.Column(sa.String(255), nullable=True)
    ticker = sa.Column(sa.String(255), nullable=True)
    factor = sa.Column(sa.String(255), nullable=True)
    memo = sa.Column(sa.String(1000), nullable=True)

class TbMacroData(StaticBase):
    """macro data"""

    __tablename__ = "tb_macro_data"
    macro_id = sa.Column(sa.ForeignKey("tb_macro.macro_id"), primary_key=True)
    trd_dt = sa.Column(sa.Date, primary_key=True)
    value = sa.Column(sa.Float, nullable=True)
    adj_value = sa.Column(sa.Float, nullable=True)


class TbMetaClass(Mixins):
    """metadata infromation"""

    __tablename__ = "tb_meta_class"
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    asset_class_cd = sa.Column(sa.String(20), nullable=True)
    asset_class_nm = sa.Column(sa.String(100), nullable=True)
    region_cd = sa.Column(sa.String(20), nullable=True)
    region_nm = sa.Column(sa.String(100), nullable=True)
    geo_cd = sa.Column(sa.String(30), nullable=True)
    geo_nm = sa.Column(sa.String(100), nullable=True)
    robo_lv1_cd = sa.Column(sa.String(10), nullable=True)
    robo_lv1_nm = sa.Column(sa.String(100), nullable=True)
    robo_lv2_cd = sa.Column(sa.String(10), nullable=True)
    robo_lv2_nm = sa.Column(sa.String(100), nullable=True)
    robo_lv3_cd = sa.Column(sa.String(10), nullable=True)
    robo_lv3_nm = sa.Column(sa.String(100), nullable=True)
    wm_lv1_cd = sa.Column(sa.String(10), nullable=True)
    wm_lv1_nm = sa.Column(sa.String(100), nullable=True)
    wm_lv2_cd = sa.Column(sa.String(10), nullable=True)
    wm_lv2_nm = sa.Column(sa.String(100), nullable=True)
    wm_lv3_cd = sa.Column(sa.String(10), nullable=True)
    wm_lv3_nm = sa.Column(sa.String(100), nullable=True)


class TbMetaUpdat(TimeSeriesBase):
    __tablename__ = "tb_meta_updat"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    aum = sa.Column(sa.Integer, nullable=True)
    trd_volume = sa.Column(sa.Integer, nullable=True)
    trd_amount = sa.Column(sa.Integer, nullable=True)


class TbTicker(Mixins):
    __tablename__ = "tb_ticker"
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    ticker_yahoo = sa.Column(sa.String(255), nullable=True)
    ticker_bloomberg = sa.Column(sa.String(255), nullable=True)
    ticker_naver = sa.Column(sa.String(255), nullable=True)
    ticker_fred = sa.Column(sa.String(255), nullable=True)
    ticker_eikon = sa.Column(sa.String(255), nullable=True)


class TbHoliday(Mixins):
    __tablename__ = "tb_holiday"
    hol_dt = sa.Column(sa.Date, primary_key=True)
    market = sa.Column(sa.VARCHAR(2), primary_key=True)
    name = sa.Column(sa.String(100), nullable=True)
    description = sa.Column(sa.String(1000), nullable=True)


class TbMLP(TimeSeriesBase):
    __tablename__ = "tb_mlp"

    trd_dt = sa.Column(sa.Date, primary_key=True)
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    win_proba = sa.Column(sa.Float)


class TbABL(TimeSeriesBase):
    __tablename__ = "tb_abl"

    trd_dt = sa.Column(sa.Date, primary_key=True)
    stk_id = sa.Column(sa.ForeignKey("tb_meta.stk_id"), primary_key=True)
    exp_rtn = sa.Column(sa.Float, nullable=True)
    r2 = sa.Column(sa.Float, nullable=True)
    equity_wgt = sa.Column(sa.Float, nullable=True)
    rate_wgt = sa.Column(sa.Float, nullable=True)


class TbLens(TimeSeriesBase):
    __tablename__ = "tb_lens"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    factor = sa.Column(sa.VARCHAR(255), primary_key=True)
    value = sa.Column(sa.Float, nullable=True)
    exp_rtn = sa.Column(sa.Float,nullable=True)
    

class TbFX(TimeSeriesBase):
    __tablename__ = "tb_fx"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    currency = sa.Column(sa.String(10), primary_key=True)
    close_prc = sa.Column(sa.Float, nullable=False)
    

class TbProbIncrease(TimeSeriesBase):
    __tablename__ = "tb_prob_increase"
    rebal_dt = sa.Column(sa.Date, primary_key=True)
    isin = sa.Column(sa.String(255), primary_key=True)
    wrap_asset_class_code = sa.Column(sa.CHAR(2), nullable=True)
    iso_code = sa.Column(sa.String(255), nullable=False)
    prob = sa.Column(sa.Numeric(9, 4), nullable=False)
    
    
class TbViewInfo(Mixins):
    __tablename__ = "tb_view_info"
    rebal_dt = sa.Column(sa.Date, primary_key=True)
    asset_class = sa.Column(sa.String(255), nullable=False)
    category = sa.Column(sa.String(255), nullable=False)
    target = sa.Column(sa.String(255), nullable=False)
    index_ticker = sa.Column(sa.String(255), primary_key=True)
    index_name = sa.Column(sa.String(255), nullable=False)
    default_weight = sa.Column(sa.Float, nullable=True)
    return_1m = sa.Column(sa.Float, nullable=True)
    avg_volume_1m = sa.Column(sa.Float, nullable=True)
    aum = sa.Column(sa.Integer, nullable=True)
    core_view = sa.Column(sa.Integer, nullable=True)
    ai_mlp_view = sa.Column(sa.Integer, nullable=True)
    ai_alpha_view = sa.Column(sa.Integer, nullable=True)
    ai_factor_view = sa.Column(sa.Integer, nullable=True)


class VwPortRiskScore(TimeSeriesBase):
    __tablename__ = "vw_port_risk_score"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    port_id = sa.Column(sa.Integer, primary_key=True)
    risk_score = sa.Column(sa.Float, nullable=True)


class TbInvstStyRtn(TimeSeriesBase):
    __tablename__ = "tb_invst_sty_rtn"
    std_dt = sa.Column(sa.Date, primary_key=True)
    port_id = sa.Column(sa.Integer, primary_key=True)
    exp_rtn = sa.Column(sa.Float, nullable=False)
    
    
class TbRegime(Mixins):
    __tablename__ = "tb_regime"
    trd_dt = sa.Column(sa.Date, primary_key=True)
    module = sa.Column(sa.String(255), primary_key=True, nullable=False)
    regime = sa.Column(sa.String(255), nullable=False)
    equity = sa.Column(sa.Float, nullable=False)
    fixed_income = sa.Column(sa.Float, nullable=False)
    alternative = sa.Column(sa.Float, nullable=False)
    liquidity = sa.Column(sa.Float, nullable=False)