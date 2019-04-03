import os
import sys , time , traceback , itertools , warnings , ipdb
import datetime as dt

from tqdm import tqdm

import pandas as pd
import pandas_datareader.data as web
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl


def backtest_intraday(lTS=None):
    windowLength = 1450#252
    _shif = 0
    foreLength = len(lTS) - windowLength
    signal = 0*lTS[-foreLength:]
    print('Running backtesting ...')
    time.sleep(1)
    for d in tqdm(range(foreLength)):

        # create a rolling window by selecting 
        # values between d+1 and d+T of S&P500 returns
        
        TS = lTS[(1+d+_shif):(windowLength+d+_shif)] 

        # Find the best ARIMA fit 
        # set d = 0 since we've already taken log return of the series
        res_tup = _get_best_model(TS)
        order = res_tup[1]
        model = res_tup[2]
        #ipdb.set_trace()
        if order is not None:
            #now that we have our ARIMA fit, we feed this to GARCH model
            p_ = order[0]
            o_ = order[1]
            q_ = order[2]

            am = arch_model(model.resid, p=p_, o=o_, q=q_, dist='StudentsT')
            res = am.fit(update_freq=5, disp='off')

            # Generate a forecast of next day return using our fitted model
            out = res.forecast(horizon=1, start=None, align='origin')

            #Set trading signal equal to the sign of forecasted return
            # Buy if we expect positive returns, sell if negative

            signal.iloc[d] = np.sign(out.mean['h.1'].iloc[-1])
        else:
            signal.iloc[d] = np.nan
    return signal




def backtest_volamodel(lTS=None,windowLength = 500):
    foreLength = len(lTS) - windowLength
    print('Running backtesting ...')
    time.sleep(1)
    try:
        for d in tqdm(range(foreLength)):

            # create a rolling window by selecting 
            # values between d+1 and d+T of S&P500 returns
            
            TS = lTS[(1+d):(windowLength+d)].copy()
            for i in range(30):
                # Find the best ARIMA fit 
                # set d = 0 since we've already taken log return of the series
                res_tup = _get_best_model(TS)
                order = res_tup[1]
                model = res_tup[2]

                if order is not None:
                    #now that we have our ARIMA fit, we feed this to GARCH model
                    p_ = order[0]
                    o_ = order[1]
                    q_ = order[2]
                    res = (arch_model(model.resid, p=p_, o=o_, q=q_, dist='StudentsT')
                            .fit(update_freq=5, disp='off')
                          )

                    #running MonteCarlo simulations

                    ipdb.set_trace()
                    # Generate a forecast of next day return using our fitted model
                    out = (res.forecast(horizon=1, start=None, align='origin')
                           .mean['h.1']
                          .iloc[-1]
                          )
                    TS = TS.append(out)

                    #Set trading signal equal to the sign of forecasted return
                    # Buy if we expect positive returns, sell if negative

    except:
        print(traceback.print_exc())
    return signal