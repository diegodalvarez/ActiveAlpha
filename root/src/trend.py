# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:32:04 2025

@author: Diego
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
from   DataCollect import DataManager

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class TrendStrats(DataManager):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        self.trend_path = os.path.join(self.data_path, "trend")
        if os.path.exists(self.trend_path) == False: os.makedirs(self.trend_path)
                
        self.lookback = 30
        
    def _generate_trend(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(signal = lambda x: x.spread.ewm(span = window, adjust = False).mean()))
        
        return df_out
        
    def generate_trend(self) -> pd.DataFrame: 
        
        df_out = (self.prep_data().groupby(
            "etf").
            apply(self._generate_trend, self.lookback).
            reset_index(drop = True))
        
        return df_out
    
    def _is_trend_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = df.dropna()
        
        df_out = (sm.OLS(
            endog = df_tmp.spread,
            exog  = sm.add_constant(df_tmp.signal)).
            fit().
            resid.
            to_frame(name = "resid").
            assign(lag_resid = lambda x: x.resid.shift()).
            merge(right = df, how = "inner", on = ["date"]).
            assign(signal_rtn = lambda x: -np.sign(x.lag_resid) * x.spread))
        
        return df_out
    
    def is_trend_rtn(self) -> pd.DataFrame: 
        
        df_out = (self.generate_trend().set_index(
            "date").
            groupby("etf").
            apply(self._is_trend_rtn).
            drop(columns = ["etf"]).
            reset_index().
            pivot(index = "date", columns = "etf", values = "signal_rtn"))
        
        path = os.path.join(self.trend_path, "is_trend.parquet")
        df_out.to_parquet(path = path, engine = "pyarrow")
        
        return df_out
        
df = TrendStrats().is_trend_rtn()
