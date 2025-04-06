# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:58:26 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
from   DataCollect import DataManager

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class CorrStrats(DataManager):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        self.corr_path = os.path.join(self.data_path, "corr")
        if os.path.exists(self.corr_path) == False: os.makedirs(self.corr_path)
        
    def _is_corr(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df[
            ["spread", "COR1M", "COR3M", "COR6M", "COR1Y"]].
            dropna())
        
        df_out = (sm.OLS(
            endog = df_tmp.spread,
            exog  = sm.add_constant(df_tmp[["COR1M", "COR3M", "COR6M", "COR1Y"]])).
            fit().
            resid.
            to_frame(name = "resid").
            assign(lag_resid = lambda x: x.resid.shift()).
            merge(right = df_tmp, how = "inner", on = ["date"]).
            assign(signal_rtn = lambda x: -np.sign(x.lag_resid) * x.spread))
        
        return df_out
        
    def is_corr(self) -> pd.DataFrame: 
        
        df_corr = (self.get_corr().assign(
            date = lambda x: pd.to_datetime(x.date)).
            pivot(index = "date", columns = "security", values = "value").
            diff().
            dropna())
        
        df_out = (self.prep_data().assign(
            date = lambda x: pd.to_datetime(x.date)).
            merge(right = df_corr, how = "inner", on = ["date"]).
            set_index("date").
            groupby("etf").
            apply(self._is_corr).
            reset_index().
            pivot(index = "date", columns = "etf", values = "signal_rtn"))
        
        out_path = os.path.join(self.corr_path, "is_corr.parquet")
        df_out.to_parquet(path = out_path)
        
        return df_out
        
CorrStrats().is_corr()