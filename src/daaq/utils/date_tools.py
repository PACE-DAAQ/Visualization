#utils/date_tools.py
import pandas as pd

def get_dates(sdate, edate, hint):
    from datetime import timedelta
    date1 = pd.to_datetime(sdate,format='%Y%m%d%H')
    date2 = pd.to_datetime(edate,format='%Y%m%d%H')
    delta = timedelta(hours=hint)
    dates = pd.date_range(start=date1, end=date2, freq=delta)
    return dates
