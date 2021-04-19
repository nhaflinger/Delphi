import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

class plotter():
    def __init__(self):
        sns.set_theme() 
   
    def simplePlot(self, data_frame, title, xval, yval, titleFontSize=12, labelFontSize=10, figureSize=(20,6), lineWidth=1.0):  
        data_frame.plot(x=xval, y=yval, ylabel=yval, legend=False, figsize=figureSize, title=title, fontsize=labelFontSize, lw=lineWidth)
        return plt

    def align_yaxis(self, ax1, v1, ax2, v2):
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny+dy, maxy+dy)

    def dualPlot(self, data_frame, title, xval, yval1, yval2, align=True, titleFontSize=12, labelFontSize=10, figureSize=(20,6), lineWidth=1.0):
        color1 = 'orangered'
        color2 = 'blue'
        fig1, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        data_frame.plot(x=xval, y=yval1, ylabel=yval1, sharey=True, ax=ax1, legend=False, figsize=figureSize, title=title, fontsize=labelFontSize, lw=lineWidth, color=color1)
        data_frame.plot(x=xval, y=yval2, ylabel=yval2, sharey=True, ax=ax2, legend=False, figsize=figureSize, title=title, fontsize=labelFontSize, lw=lineWidth, color=color2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylabel(yval1, fontsize=labelFontSize, color=color1)
        ax2.set_xlabel(xval, fontsize=labelFontSize)
        ax2.set_ylabel(yval2, fontsize=labelFontSize, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        if (align):
            self.align_yaxis(ax1, 0, ax2, 0)
        return plt

    def multiPlot(self, data_frame, title, xval, yval_list, titleFontSize=12, labelFontSize=10, figureSize=(20,6), lineWidth=1.0, sizeLimit=100):
        fig1 = data_frame.plot(x=xval, y=yval_list[:sizeLimit], figsize=figureSize, title=title, fontsize=labelFontSize, lw=lineWidth)
        return plt
   
    def stackedPlot(self, data_frame, title, xval, yval_list, titleFontSize=12, labelFontSize=10, figureSize=(20,6), lineWidth=1.0, sizeLimit=100):
        axes = data_frame.plot(x=xval, y=yval_list[:sizeLimit], subplots=True, sharex=True, sharey=True, legend=False, figsize=figureSize, title=title, fontsize=labelFontSize, lw=lineWidth)
        idx = 0
        for yval in yval_list[:sizeLimit]:
            axes[idx].set_ylabel(yval, fontsize=labelFontSize)
            axes[idx].set_xlabel(xval, fontsize=labelFontSize)
            idx += 1    
        return plt
   
    def autocorrelationPlot(self, data_frame, srange=(0,40)):
        labels = []
        colors = []
        cmap = plt.get_cmap("tab10")
        idx = 0
        for col in data_frame.columns:
            series = data_frame[col].iloc[srange[0]:srange[1]]
            ax = autocorrelation_plot(series, color=cmap(idx))
            colors.append(cmap(idx))
            idx += 1
            labels.append(col)

        custom_lines = []
        for col in colors:
            custom_lines.append(Line2D([0], [0], color=col, lw=4))
        ax.legend(custom_lines, labels, loc='upper right')    

    def VARforecastPlot(self, data_frame, model, steps=5):
        lag_order = model.k_ar
        output = model.forecast(data_frame.values[-lag_order:], steps)
        model.plot_forecast(steps)
        return output

    def ARIMAresidualsPlot(self, data_frame, fits):
        sns.set_theme() 
        for col in data_frame.columns:
            index = data_frame.index
            if (index.name == col):
                continue
            residuals = pd.DataFrame(fits[col].resid)
            residuals.plot(title=col, legend=False)
        return residuals


    def VARMAXforecastPlot(self, endog, train_fraction=0.66, steps=1, order=(2,0), maxiter=1000, verbose=False, error_cov_type='diagonal', title='', titleFontSize=12, labelFontSize=10, figureSize=(20,6), lineWidth=1.0):
        sns.set_theme()         
        
        cleaned = endog.dropna()
        size = int(len(cleaned) * train_fraction)
        timestamp = cleaned.index.to_timestamp().values.tolist()
        train = cleaned[0:size].to_numpy().tolist()
        test = cleaned[size:len(cleaned)].to_numpy().tolist()

        history = [z for z in train]
        predictions = train
        interval = 10
        for t in range(len(test)):
            model = VARMAX(history, order=order, error_cov_type=error_cov_type)
            model_fit = model.fit(maxiter=maxiter, disp=False)
            output = model_fit.forecast(steps)
            yhat = output[0].tolist()
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            if (t % 10 == 0 and verbose):
                print('percent done: ' + str(t/len(test)))
        print('percent done: 1.00')

        if (len(endog.dropna().to_numpy().tolist()) != len(predictions)):
            print('List lengths do not match')
        
        col = 0
        for val in endog:  
            plt.subplots(figsize=figureSize) 
            plt.title(title, fontsize=titleFontSize)
            plt.xlabel('Timestamp', fontsize=labelFontSize)
            plt.ylabel(endog[val].name, fontsize=labelFontSize)
            column = [row[col] for row in predictions]
            plt.plot(timestamp, column, color='red', linewidth=lineWidth, linestyle='--')
            col += 1
            plt.plot(timestamp, endog[val].dropna().to_numpy().tolist(), linewidth=lineWidth)
            plt.legend(labels=('prediction', endog[val].name))
            plt.show()

        return predictions
        
    def ARIMAforecastPlot(self, xval, yval, train_fraction=0.66, order=(5,1,0), title='', titleFontSize=12, labelFontSize=10, figureSize=(20,6), lineWidth=1.0):
        sns.set_theme()   
        size = int(len(yval) * train_fraction)
        timestamp = xval.dropna().to_numpy().tolist()
        train = yval[0:size].dropna().to_numpy().tolist()
        test = yval[size:len(yval)].to_numpy().tolist()
        history = [x for x in train]
        predictions = train 
        for t in range(len(test)):
            model = ARIMA(history, order=order)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

        if (len(yval.dropna().to_numpy().tolist()) != len(predictions)):
            print('List lengths do not match')

        plt.subplots(figsize=figureSize)    
        plt.title(title, fontsize=titleFontSize)
        plt.xlabel(xval.name, fontsize=labelFontSize)
        plt.ylabel(yval.name, fontsize=labelFontSize)
        plt.plot(timestamp, predictions, color='red', linewidth=lineWidth, linestyle='--')
        plt.plot(timestamp, yval.dropna().to_numpy().tolist(), linewidth=lineWidth)
        plt.legend(labels=('prediction', yval.name))

        return predictions