import numpy as np
import pandas as pd
import cvxpy as cp

def opt(w, p, portfolio_value=3000000, min_shares=1):
        w = w.values
        p = p.values
        n = len(w)
        portfolio_value = portfolio_value
        min_shares = min_shares 

        shares = cp.Variable(n, integer=True)
        cash = portfolio_value - p @ shares

        u = cp.Variable(n)
        eta = w * portfolio_value - cp.multiply(shares, p)

        _obj = cp.sum(u) + cash

        _cons = [
            eta <= u,
            eta >= -u,
            cash >= 0,
            shares >= min_shares,
        ]

        _opt = cp.Problem(cp.Minimize(_obj), _cons)

        _opt.solve(verbose=False)

        return shares.value

def calc_table():
    row_num = len(df)
    df['평가금액'][:-2] = df['현재가'][:-2]*df['수량'][:-2]
    df['평가금액'][row_num-2] = money - np.sum(df['평가금액'][:-2])
    df['실잔고비중'][:-1] = df['평가금액'][:-1]/money
    df['괴리율'][:-1] = abs(df['실잔고비중'][:-1]-df['목표비중'][:-1])
    df['평가금액'][row_num-1] = np.sum(df['평가금액'][:-1])
    df['실잔고비중'][row_num-1] = np.sum(df['실잔고비중'][:-1])
    df['괴리율'][row_num-1] = np.sum(df['괴리율'][:-1])

money = 1000000

data = {'isin':['KR7239660004','KR7114470008','KR7214980005','KR7153130000','KR7133690008','KR7294400007','KR7305540007',
                    'KR7152100004','KR7329670004','KR7329660005'],
        '자산군':['채권','채권','현금형','현금형','주식','주식','주식','주식','대체',
                    '대체'],
        '현재가':[112195, 108205, 107195, 106520, 88870, 42650, 36345, 34850, 14160, 12745],
        '목표비중':[0.025,0.025,0.025,0.025,0.175,0.175,0.175,0.175,0.1,0.1] #level 5
        # '목표비중':[0.05,0.05,0.15,0.15,0.1,0.1,0.1,0.1,0.1,0.1] #level 4
        # '목표비중':[0.025,0.025,0.3,0.3,0.075,0.075,0.075,0.075,0.025,0.025] # level 3
}
df = pd.DataFrame(data).set_index("isin")

w = df['목표비중']
p = df['현재가']

shares = opt(w=w, p=p, portfolio_value=money)

df['수량'] = shares

row_num = len(df)
df.insert(4,'평가금액',0)
df.insert(5,'실잔고비중',0)
df.insert(6,'괴리율',0)

df.loc["현금"] = {"자산군":"", "현재가":0, "목표비중":0, "수량":0}
df.loc["합계"] = {"자산군":"", "현재가":0, "목표비중":1, "수량":0}

new_column_order = ["자산군", "현재가", "수량", "평가금액", "실잔고비중", "목표비중", "괴리율"]
df = df[new_column_order]
calc_table()
df.to_clipboard()
df