
import urllib , xmltodict , time 
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import warnings
from matplotlib import animation
import pandas_datareader.data as web
from tqdm import tqdm

#polinomial fitting

def _fit_poly(tresdf):
    x = np.array([-2.0,0.0,2.0])
    xp = np.linspace(-3, 3, 100)
    curve_params = pd.DataFrame()

    curve_data = pd.DataFrame(index=xp)
    print('Solving polynomials ...')
    time.sleep(1)
    for i in tqdm(range(len(tresdf))):
        try:
            y = np.array(tresdf.iloc[i])
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            px = p(xp)
            #_ = plt.plot(x, y, '.', xp, p(xp), '-')
            #time.sleep(2)
            #plt.ylim(-3,3)
        except:
            z = np.repeat(None,4)
            px = np.repeat(None,100)
        #curve_data[tresdf.index[i].strftime('%Y-%m-%d')] = px
        curve_data[tresdf.index[i]] = px
        curve_params = curve_params.append(pd.Series(z,index=['pow3','pow2','pow1','c'],name=tresdf.index[i])
                                              # ,ignore_index=True
                                              )
    curve_params.index = pd.to_datetime(curve_params.index)
    curve_params = curve_params.dropna()
    curve_data = curve_data.dropna(axis=1)
    curve_data.columns = [x.split('T')[0] for x in curve_data.columns]
    return curve_params , curve_data

def plot_multi(data, cols=None,title='', spacing=.1,loc='upper center', **kwargs):

    from pandas import plotting
    plt.figure()
    plt.title(title)
    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_style'), '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_xlabel("Days")
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)])
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=loc
              #, bbox_to_anchor=(1, 0.5)
             )
    return ax

