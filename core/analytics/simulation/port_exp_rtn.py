import os
import sys
import logging
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))
from hive import db
from config import get_args

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)

# logger.info(f"running DWS expected return {TODAY:%Y-%m-%d}")

class PortExpectedReturn:
    """Portfolio Expected Return Class
    """

    def risk_level_conditions(self, row):
        """For the DWS strategy, this function returns the risk level based on the port ID.

        Args:
            row (pd.DataFrame): DataFrame which contains port id

        Returns:
            int: risk level
        """
        if (row['port_id']>=21) & (row['port_id'] <=25):
            val = 1
        elif (row['port_id']>=26) & (row['port_id'] <=30):
            val = 2
        elif (row['port_id']>=31) & (row['port_id'] <=35):
            val = 3
        elif (row['port_id']>=36) & (row['port_id'] <=40):
            val = 4
        elif (row['port_id']>=41) & (row['port_id'] <=45):
            val = 5
        elif (row['port_id']>=46) & (row['port_id'] <=50):
            val = 1
        elif (row['port_id']>=51) & (row['port_id'] <=55):
            val = 2
        elif (row['port_id']>=56) & (row['port_id'] <=60):
            val = 3
        elif (row['port_id']>=61) & (row['port_id'] <=65):
            val = 4
        elif (row['port_id']>=66) & (row['port_id'] <=70):
            val = 5
        return val

    # Risk Level별 BM 수익율(annual return) 및 변동성(vol) 산출
    def get_bm_ar_vol(self, year):
        """This Function returns bench mark annual returns and volatilities based on risk level.
        Bench Mark Ticker: LEGATRUU Index, MXWD Index, SPGSCITR Index, BIL

        Args:
            year (int): How many years ago does the data start?

        Returns:
            pd.DataFrame: bench mark annual returns and volatilities
        """
        querystr = "trd_dt > '"+ (datetime.now()-timedelta(days=int(year*365.25))).strftime("%Y-%m-%d") + "' "
        tickers=['LEGATRUU Index', 'MXWD Index', 'SPGSCITR Index']
        macro_data = db.get_macro_data_from_ticker(tickers, datetime.now()-timedelta(days=int(year*365.25))).sort_index()
        fx= db.get_fx('USD').sort_index()
        BIL_daily_price=  db.get_price('BIL').query(querystr).sort_index()
        daily_price = pd.merge(left = BIL_daily_price , right = macro_data , how = 'inner', on = 'trd_dt')
        daily_price = pd.merge(left = daily_price, right = fx , left_index=True, right_index=True, how = 'inner')
        for i in range(4):
            daily_price.iloc[:,i]=daily_price.iloc[:,i].multiply(daily_price.iloc[:,5])
        daily_price = daily_price.drop(columns = ['currency','close_prc'])
        daily_returns = daily_price.sort_index().pct_change().dropna()
        daily_returns.columns=['liquidity', 'fixedincome', 'equity', 'alternative']
        n_data_per_year = int((len(daily_returns)-1)/year)
        yearly_returns= (1 + daily_returns).cumprod(axis=0).iloc[::n_data_per_year, :].pct_change().dropna()

        weights = pd.DataFrame({'equity': [0.0, 0.1, 0.3, 0.4, 0.7]
                        , 'fixedincome': [0.0, 0.1, 0.05, 0.1, 0.05]
                        , 'alternative': [0.0, 0.1, 0.05, 0.2, 0.2]
                        , 'liquidity': [1.0, 0.7, 0.6, 0.3, 0.05]}, 
                        index = [1, 2, 3, 4, 5])
        w=0.3
        n=3
        mix_annual_returns = (1-w)* yearly_returns.iloc[:-n].mean() + w*yearly_returns.iloc[-n:].mean()
        mix_annual_returns.name='bm_annual_returns'
        bm_ar = pd.DataFrame(weights@mix_annual_returns, columns=['bm_annual_returns'])
        bm_ar['risk_level']=bm_ar.index

        division_point = n_data_per_year*n
        Sigma1 = daily_returns.iloc[:-division_point].cov()*252*(1-w)
        Sigma2 = daily_returns.iloc[-division_point:].cov()*252*w
        Sigma= Sigma1 + Sigma2

        bm_vol=[]
        for i in range(len(weights)):
            bm_vol.append(np.sqrt(weights.iloc[i,:].T@Sigma@weights.iloc[i,:]))

        bm_vol =pd.DataFrame(bm_vol,index=[1, 2, 3, 4, 5], columns=['bm_vol'])
        bm_vol['risk_level'] = bm_vol.index

        bm = pd.merge(left=bm_vol, right= bm_ar, how= 'inner', on ='risk_level')
        return bm

    # tb_port_value 에서 값을 가지고 와서, port 별 수익율과 변동성 산출
    def get_port_ar_vol(self, port_id, year):
        """This Function returns portfolio annual returns and volatilities.

        Args:
            port_id (int): portfolio id
            year (int): How many years ago does the data start?

        Returns:
            pd.DataFrame: portfolio annual returns and volatilities
        """
        port_value = db.get_port_value(port_id)
        fx= db.get_fx('USD').sort_index()
        port_value = pd.merge(left = port_value, right = fx , left_index=True, right_index=True, how = 'inner')
        # US port의 경우, 환율 적용
        for i in range(25):
            port_value.iloc[:,25+i]=port_value.iloc[:,25+i].multiply(port_value.iloc[:,51])
        port_value = port_value.drop(columns = ['currency','close_prc'])

        # 무위험 이자율 데이터
        riskfree_ticker=['USGG3M Index']
        rf= db.get_macro_data_from_ticker(riskfree_ticker, datetime.now()-timedelta(days=int(year*365.25)))
        rf.columns=['risk_free_rate']
        # 1영업일 무위험 이자율 계산
        rf.iloc[:,0]=(1+rf.iloc[:,0]*0.01)**(float(1.0/252.0))-1
        rf.index = pd.to_datetime(rf.index)

        re= port_value.sort_index().pct_change()
        re = pd.merge(left = re, right = rf , left_index=True, right_index=True, how = 'inner')
        # 무위험 이자율을 빼준 수익율(re2) 따로 저장
        re2=re.copy()
        for i in range(50):
            re2.iloc[:,i]=re.iloc[:,i]-re.iloc[:,50]

        port_total_return = (1 + re2).cumprod(axis=0).iloc[-1]
        port_ar = (port_total_return ** (float(252.0/len(re2)))) - 1
        port_ar.name='port_annual_return'
        re=re.drop(columns = ['risk_free_rate'])

        # port별 변동성 산출
        port_vol = re.std()*(252 ** 0.5)
        port_vol=pd.DataFrame(port_vol,columns=['port_vol'])

        port = pd.merge(left=port_vol, right= port_ar, how= 'inner', left_index=True, right_index=True)
        return port

    #port와 bm의 정보를 바탕으로 port별 기대 수익율 계산
    def get_port_exp_return(self, port, bm):
        """Calculate expected rate of return for each port based on port and BM information

        Args:
            port (pd.DataFrame): Portfolio annual returns and volatilities
            bm (pd.DataFrame): Bench Mark annual returns and volatilities

        Returns:
            pd.DataFrame: expected rate of return for each port
        """
        port['port_id']=port.index
        port['risk_level']= port.apply(self.risk_level_conditions, axis=1)
        port = pd.merge(left=port, right= bm, how= 'inner', on ='risk_level')
        port['vol_diff']= np.where(port['port_vol']-port['bm_vol'] > 0 , port['port_vol'] - port['bm_vol'], 0)
        port['port_exp_return'] = port['bm_annual_returns'] + port['vol_diff'] * port['port_annual_return']/port['port_vol']
        port=port.sort_index()
        return port

    # port별 기대 수익율이 계산되면, TB_INVST_STY_RTN에 insert
    def insert_to_TbInvstStyRtn(self, port):
        """This function insert data into tb_invststyrtn DB table

        Args:
            port (pd.DataFrame): expected rate of return for each port
        """
        port_exp_rtn= port[['port_exp_return','port_id']].copy()
        port_exp_rtn['std_dt'] = datetime.now()
        port_exp_rtn.columns=['exp_rtn','port_id','std_dt']
        try:
            db.TbInvstStyRtn.insert(port_exp_rtn)
        except:
            db.TbInvstStyRtn.update(port_exp_rtn)


    def run_port_exp_rtn(self, year = 10):
        """Exacuting function for portfolio expected return calculation

        Args:
            year (int, optional): How many years ago does the data start? Defaults to 10.

        """
        extra = dict(user=args.user, activity="update dws portfolio expected return", category="script")
        # year: BM의 과거 데이터 관찰 기간
        #월요일에만 실행
        if datetime.today().weekday() != 0:
            logger.info(msg=f"[SKIP] DWS expected return. {TODAY:%Y-%m-%d}", extra=extra)
            return
        
        logger.info(msg=f"[PASS] Start DWS expected return update. {TODAY:%Y-%m-%d}", extra=extra)
        # DWS port_id
        port_id = list(range(21,71))
        # Risk Level별 BM 수익율(annual return) 및 변동성(vol) 산출
        bm = self.get_bm_ar_vol(year)
        # tb_port_value 에서 값을 가지고 와서, port 별 수익율과 변동성 산출
        port = self.get_port_ar_vol(port_id, year)
        #port와 bm의 정보를 바탕으로 port별 기대 수익율 계산
        port = self.get_port_exp_return(port, bm)
        # port별 기대 수익율이 계산되면, TB_INVST_STY_RTN에 insert
        self.insert_to_TbInvstStyRtn(port)
        logger.info("[PASS] End DWS exp_rtn update.")
