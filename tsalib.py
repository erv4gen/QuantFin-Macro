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





def variance_quantiles(sim_paths,prc=0.75):
    
    sims_vars = []
    for i in range(len(sim_paths)):
        sims_vars.append(np.var(sim_paths[i]))
    
    qa = np.quantile(sims_vars,q=[0.25,prc,0.95])
    sims_vars_np = np.array(sims_vars)
    return sim_paths[sims_vars.index( np.sort(sims_vars_np[(sims_vars_np > qa[1]) & (sims_vars_np < qa[2]) ])[0] )]

def find_arch(TS,model,true_variance_forecast,horizon,print_model):

    
    aic_d = []
    RMSE = []
    params_factor  = {'p':list(range(1,5)), 'o':list(range(1,5))}
    params_factor_list = [item for key, item  in params_factor.items()]
    params_factor_grid = list(itertools.product(*params_factor_list))
    print(f'Solving best {model} model ...')
    time.sleep(1)
    for param in tqdm(params_factor_grid):
        p_ = param[0]
        q_ = param[1]
        o_ = 0
        #test each parameter on three different month
        eval_aic = []
        eval_rmse = []
        for step in range(88,0,-22):
            res = (arch_model(TS[:-step], p=p_, o=o_, q=q_, dist='StudentsT',vol='GARCH')
                                .fit(update_freq=5, disp='off')
                              )
            forecast  = (res.forecast(horizon=horizon, start=None, align='origin', method='simulation')
                    )
            expected_variance =  forecast.variance.iloc[-1].values
            eval_aic.append(res.aic)
            #ipdb.set_trace()
            if step == 22:
                eval_rmse.append( np.sqrt(np.mean(np.power(expected_variance - true_variance_forecast[-step:],2) )) )
            else:
                eval_rmse.append( np.sqrt(np.mean(np.power(expected_variance - true_variance_forecast[-step:(step*(-1)+horizon)],2) )) )
       
        aic_d.append(np.product(np.log(eval_aic) ) )
        RMSE.append(np.product(eval_rmse))
    combo_param = [a*b*1e3 for a,b in zip(aic_d ,RMSE)]
    best_params= params_factor_grid[(combo_param.index(min(combo_param)))]
    
    if print_model:
        fig, ax = plt.subplots(1,1)
        title = plt.title('AIC Estimation')
        lines = ax.plot(aic_d[1:],label='AIC',color='#a82828')
        #lines[0].set_label('AIC')

        ax2 = ax.twinx()
        lines2 = ax2.plot(RMSE,label='RMSE',color='#6d9686')
        #lines2[0].set_label('RMSE')

        legend = ax.legend()
        legend2 = ax2.legend()
    
    
    print('best parameters: ',best_params, 'At index:',params_factor_grid.index(best_params))
    return params_factor_grid , best_params

def vola_estimation(data,horizon,model,print_model,sim_prc):
    TS = 100* data['Adj Close'].pct_change().dropna()
    #TS = rets[:-horizon]
    #predict_date = 
    params_factor_grid , best_params = find_arch(TS=TS[:-horizon]
                                                 ,model=model
                                                 ,true_variance_forecast=data.vola.values#[-horizon:].values
                                                 ,horizon=horizon
                                                 ,print_model=print_model)
    
    res_final = (arch_model(TS, p=best_params[0], o=0, q=best_params[1], dist='StudentsT',vol=model)
                                .fit(update_freq=5, disp='off')
            )


    forecast  = (res_final.forecast(horizon=horizon, start=None, align='origin', method='simulation')
            )
    
    sims = forecast.simulations

    simulated_path = sims.residual_variances[-1,:] #/100
    expected_variance =  forecast.variance.iloc[-1].values #/100                              

    most_var_sim_ts = variance_quantiles(simulated_path,prc=sim_prc)
    
    month_expect_var = np.sum([np.power(x-most_var_sim_ts.mean(),2) for x in most_var_sim_ts]) *100/ data.iloc[-horizon,1]
    
    return {'expected_var':month_expect_var,'expected_var_ts':most_var_sim_ts}