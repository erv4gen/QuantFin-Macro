import os
import sys , time , traceback 
import datetime as dt

from tqdm import tqdm
import warnings , ipdb
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


def tsplot(y, lags=None, figsize=(8, 6), style='ggplot',title='Time Series Analysis Plots'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


#Create backtester

# Find Arima model

def get_best_arma(TS):
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    print('Solving arima model ...')
    time.sleep(1)
    for i in tqdm(pq_rng):
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except Exception as e:
                    #print(e)
                    continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl


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


def variance_quantiles(sim_paths,prc=0.75):
    sims_vars = []
    for i in range(len(sim_paths)):
        sims_vars.append(np.var(sim_paths[i]))
    qa = np.quantile(sims_vars,q=[0.25,0.5,prc])
    sims_vars_np = np.array(sims_vars)
    return sim_paths[sims_vars.index((sims_vars_np[sims_vars_np > qa[2]][0]))]




