# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:21:55 2025

@author: Diego
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 19:24:54 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

class DataManager:
    
    def __init__(self) -> None:
        
        self.dir       = os.path.dirname(os.path.abspath(__file__))  
        self.root_path = os.path.abspath(
            os.path.join(os.path.abspath(
                os.path.join(self.dir, os.pardir)), os.pardir))
        
        self.data_path      = os.path.join(self.root_path, "data")
        self.raw_data_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_data_path) == False: os.makedirs(self.raw_data_path)
        
        self.bbg_path    = r"C:\Users\Diego\Desktop\app_prod\BBGData\data"
        self.bbg_tickers = ["COR1M", "COR1Y", "COR3M", "COR6M"] 
        
        self.ticker_path = os.path.join(self.raw_data_path, "tickers.csv")
        self.df_tickers  = pd.read_csv(filepath_or_buffer = self.ticker_path)
        
    def get_yf(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.raw_data_path, "YFETFs.parquet")
        try:
            
            if verbose == True: print("Seaching for YF ETFs")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            
            if verbose == True: print("Couldn't find it, collecting it now") 
        
            tickers = (self.df_tickers.melt(
                id_vars = ["name"]).
                value.
                drop_duplicates().
                to_list())
        
            start_date = dt.date(year = 1990, month = 1, day = 1)
            
            df_out     = (pd.concat([
                yf.Ticker(ticker).history(auto_adjust = False, start = start_date).assign(ticker = ticker)
                for ticker in tickers]).
                reset_index().
                rename(columns = {
                    "Date"     : "date",
                    "Close"    : "close",
                    "Adj Close": "adj_close"}).
                assign(
                    ticker = lambda x: x.ticker.str.replace("^", ""), 
                    date   = lambda x: pd.to_datetime(x.date).dt.date)
                [["date", "close", "adj_close", "ticker"]])
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def get_corr(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "Corr.parquet")
        try:
            
            if verbose == True: print("Seaching for Implied Correlations")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting") 
        
            paths = [os.path.join(
                self.bbg_path, ticker + ".parquet")
                for ticker in self.bbg_tickers]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                drop(columns = ["variable"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                etf_rtn   = lambda x: x.etf_px.pct_change(),
                bench_rtn = lambda x: x.bench_px.pct_change(),
                spread    = lambda x: x.etf_rtn - x.bench_rtn))
        
        return df_out
    
    def prep_data(self) -> pd.DataFrame: 
    
        df_tmp = self.get_yf().drop(columns = ["close"])
        
        df_etf = (df_tmp.rename(
            columns = {
                "adj_close": "etf_px",
                "ticker"   : "etf"}))
        
        df_benchmark = (df_tmp.rename(
            columns = {
                "adj_close": "bench_px",
                "ticker"   : "benchmark"}))
        
        df_out = (df_etf.merge(
            right = self.df_tickers, how = "inner", on = ["etf"]).
            merge(right = df_benchmark, how = "inner", on = ["date", "benchmark"]).
            groupby(["etf", "benchmark"]).
            apply(self._get_rtn).
            reset_index(drop = True))
        
        return df_out
    
def main() -> None:
            
    DataManager().get_yf(verbose = True)
    DataManager().get_corr(verbose = True)
    
if __name__ == "__main__": main()