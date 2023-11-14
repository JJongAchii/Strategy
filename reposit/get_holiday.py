import pandas as pd
from datetime import date, timedelta
import holidays


check_date = date.today()
# end_date = check_date + timedelta(days=30)
check_date = check_date - timedelta(days=11060)
us_holidays = holidays.KR()
holidays_list = []

for i in range(20000):
    check_date = check_date + timedelta(days=1)

    if check_date in us_holidays:
        holidays_list.append(check_date)


df = pd.DataFrame(holidays_list, columns=['date'])
df['market'] = 'KR'
df.to_clipboard()

