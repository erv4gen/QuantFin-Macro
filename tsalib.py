import os
import sys , time , traceback , itertools , warnings , ipdb , uuid
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

warnings.filterwarnings('ignore')

def get_data(date,ticker,data_source,horizon):
    if data_source=='web':
        start = '2000-01-01'
        end = date
        end_horizon =dt.datetime.strftime(dt.datetime.strptime(end,'%Y-%m-%d') + dt.timedelta(days=horizon)
                             ,'%Y-%m-%d')

        get_px = lambda x: web.DataReader(x, 'yahoo', start=start, end=end)

        # symbols = ['SPY','TLT','MSFT']
        # # raw adjusted close prices
        # data = pd.DataFrame({sym:get_px(sym)['Adj Close'] for sym in symbols})

        ticker_data = get_px(ticker)
        ticker_data['vola'] = 100*(ticker_data.High - ticker_data.Low)/ticker_data['Adj Close']

        ticker_data['rets'] = 100* ticker_data['Adj Close'].pct_change()
        ticker_data = ticker_data.dropna()
    else:
        pass
            #ticker_data = get_px(ticker)
        # log returns
        #lrets = np.log(ticker_data['Adj Close']/ticker_data['Adj Close'].shift(1)).dropna()
        
    return ticker_data


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

def find_arch(TS,model,true_variance_forecast,horizon,print_model,method,dist):

    
    aic_d = []
    RMSE = []
    params_factor  = {'p':list(range(1,5)), 'o':list(range(1,5))}
    params_factor_list = [item for key, item  in params_factor.items()]
    params_factor_grid = list(itertools.product(*params_factor_list))
    print(f'Solving best {model} model ...')
    time.sleep(1)
    
    for param in tqdm(params_factor_grid):
        #ipdb.set_trace()
        p_ = param[0]
        q_ = param[1]
        o_ = 0
        #test each parameter on three different month
        eval_aic = []
        eval_rmse = []
        for step in range(88,0,-22):
            res = (arch_model(TS[:-step], p=p_, o=o_, q=q_, dist=dist,vol=model)
                                .fit(update_freq=5, disp='off')
                              )
            forecast  = (res.forecast(horizon=horizon, start=None, align='origin', method=method)
                    )
            expected_variance =  forecast.variance.iloc[-1].values
            eval_aic.append(res.aic)
            #ipdb.set_trace()
            if step == 22:
                eval_rmse.append( np.sqrt(np.mean(np.power(expected_variance - true_variance_forecast[-step:],2) )) )
            else:
                eval_rmse.append( np.sqrt(np.mean(np.power(expected_variance - true_variance_forecast[-step:(step*(-1)+horizon)],2) )) )
       
        aic_d.append(np.mean(eval_aic ) )
        RMSE.append(np.mean(eval_rmse ))
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

def build_model(data,horizon,model,print_model,sim_prc,method,arch_params,mean_model,dist):
    #TS = 100* data['Adj Close'].pct_change().dropna()
    #TS = rets[:-horizon]
    #predict_date = 
    if arch_params is None:
        params_factor_grid , best_params = find_arch(TS=data.rets[:-horizon]
                                                 ,mean=mean_model
                                                 ,true_variance_forecast=data.vola.values#[-horizon:].values
                                                 ,horizon=horizon
                                                 ,print_model=print_model
                                                 ,model=model
                                                 ,dist=dist
                                                 ,method=method)
    else:
        best_params = arch_params
    res_final = (arch_model(data.rets, p=best_params[0], o=0, q=best_params[1], dist=dist,vol=model)
                                .fit(update_freq=5, disp='off')
            )


    forecast  = (res_final.forecast(horizon=horizon, start=None, align='origin', method=method)
            )
    if method=='simulation' or method=='bootstrap':
        sims = forecast.simulations

        #simulated_path_var = sims.residual_variances[-1,:] #/100
        simulated_path_var = sims.values[-1,:]#.iloc[-1].values #/100
        var_sim_ts = variance_quantiles(simulated_path_var,prc=sim_prc)
        
    else:
        var_sim_ts = np.zeros(horizon)
        
    #ipdb.set_trace()
    #expected_mean = forecast.mean.iloc[-1].values#.iloc[-1].values #/100
    expected_variance =  forecast.variance.iloc[-1].values #/100                              
    
    month_expect_var_d = np.sqrt(np.sum([np.power(x-var_sim_ts.mean(),2) for x in var_sim_ts]) *100/21)#/ data.iloc[-horizon,2]
    month_expect_var = np.sqrt(np.sum([np.power(x-expected_variance.mean(),2) for x in expected_variance]) *100)
#     month_expect_var =  np.sqrt( np.mean(
#                                         #np.sum(
#             [np.power(x-expected_mean,2) for x in expected_mean]
#                                         #)      
#                                         )
#                                 )
    
    if print_model:
        fig, ax = plt.subplots(1,1)
        if method=='simulation' or method=='bootstrap':
            lines = plt.plot(simulated_path_var[::30].T, color='#9cb2d6')
            lines[0].set_label('Simulated path')
        line = plt.plot(expected_variance, color='#002868')
        line[0].set_label('Expected variance')
        legend = plt.legend()
        
        f = res_final.hedgehog_plot(type='mean')
        print(res_final)

    
    return {'expected_var':month_expect_var,'expected_var_d':month_expect_var_d#, 'values':expected_mean
           }

def eval_model(TS,sim_prc,print_model,model,horizon,method,arch_params,mean_model,dist,ticker):
    prev_month_vola =  np.sqrt( np.sum(
            [np.power(x-TS.rets[-(horizon*2):-horizon].values.mean(),2) for x in TS.rets[-(2*horizon):-horizon].values]
                                                            ) )

    var_sim = build_model(data=TS.iloc[:-horizon]
                                     ,horizon=horizon
                                     ,model=model
                                     ,print_model=print_model
                                     ,sim_prc=sim_prc
                                    ,method=method
                                      ,dist=dist
                                     ,arch_params=arch_params
                                     ,mean_model=mean_model) 

    real_vola =  np.sqrt( np.sum(
            [np.power(x-TS.rets[-horizon:].values.mean(),2) for x in TS.rets[-horizon:].values]
                                                            ) )
    return {'model': 
                    {
                        'sim_id': str(uuid.uuid4())
                        ,'ticker':ticker
                        ,'arch_model':model
                    ,'sim_prc':sim_prc
                    ,'method':method
                    ,'dist':dist
                    ,'arch_params':arch_params
                    ,'mean_model':mean_model
                    ,'date': dt.datetime.strftime(TS.index[-1],'%Y-%m-%d')
                        ,'prev_month_vola': round(prev_month_vola,2)
                       # , 'expected_var':round(var_sim['expected_var'],2)
                    , 'expected_var_d':round(var_sim['expected_var_d'],2)
                       ,'real_vola':round(real_vola,2)
                    }
            ,'stats':
                    {'null_rmse' : round( np.abs(prev_month_vola- real_vola) / prev_month_vola
                                           ,2)
                   #  ,'arch_rmse' : round(np.abs(var_sim['expected_var']- real_vola) / var_sim['expected_var'] ,2 )
                     ,'arch_sim_rmse' : round(np.abs(var_sim['expected_var_d']- real_vola) / var_sim['expected_var_d'] ,2 )
                    }
           }


def test_prediction(n_month_back,date,ticker,sim_prc,data_source,model,print_model,method,arch_params,mean_model,dist):
    horizon=22
    ticker_data = get_data(date,ticker,data_source,horizon=horizon)
    res_dict = []
    by_month_dict =[]
    print('Running backtest ...')
    time.sleep(1)
    for step in tqdm(range(22*n_month_back,0,-22)):
        res_dict.append(eval_model(TS=ticker_data.iloc[:-step]
                                   ,ticker=ticker
                                   ,sim_prc=sim_prc
                                   ,print_model=print_model
                                   ,model=model
                                   ,horizon=horizon
                                  ,method=method
                                   ,dist=dist
                                   ,mean_model=mean_model
                                  ,arch_params=arch_params))
        
        #by_month_dict.append(res_dict)
    null_error= round(sum(d['stats']['null_rmse'] for d in res_dict) / len(res_dict),2)
   # arch_error = round(sum(d['stats']['arch_rmse'] for d in res_dict) / len(res_dict),2)
    arch_d_error = round(sum(d['stats']['arch_sim_rmse'] for d in res_dict) / len(res_dict),2)
    
    return {'by_month':res_dict
            ,'summary':{'null_rmse':null_error
                     #   ,'arch_rmse':arch_error
                        ,'arch_sim_rmse':arch_d_error
                       }}