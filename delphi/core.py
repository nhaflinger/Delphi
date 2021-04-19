import os
import glob
import re
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VARMAX
from scipy.stats import pearsonr
from tensorflow import keras
from tensorflow.keras import layers
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

class core():
    def __init__(self):
        self.data_frame = pd.DataFrame

    def readCSV(self, file_name):
        if (os.path.exists(file_name)):
            self.data_frame = pd.read_csv(file_name)
        return self.data_frame

    def readTable(self, file_name):
        if (os.path.exists(file_name)):
            self.data_frame = pd.read_table(file_name)
        return self.data_frame
        
    def readExcel(self, file_name):
        if (os.path.exists(file_name)):
            self.data_frame = pd.read_excel(file_name)
        return self.data_frame

    def setDataFrame(self, data_frame):
        self.data_frame = data_frame

    def getFileList(self, directory, pattern):    
        # get all files in directory
        file_names = []
        if (os.path.exists(directory)):
            file_names = glob.glob(directory + "/" + pattern + "*.csv")
        return file_names

    def seriesSearch(self, regex):      
        yval_list = list(self.data_frame.head(0))
        new_list = []
        for item in yval_list:
            if (re.search(regex, item)):
                new_list.append(item)
        return new_list

    def getDataFrames(self, file_names):   
        # return list of data frames
        dfs = []
        for file_name in file_names:
            dfs.append(readCSV(file_name))
        return dfs

    def setIndex(self, index):
        dateTimeIndex = self.data_frame.set_index(index)
        self.data_frame = dateTimeIndex

    def toDateTime(self, index):
        self.data_frame[index] = pd.to_datetime(self.data_frame[index], unit='s')

    def groupBy(self, key, freq):   
        grouper = self.data_frame.groupby(pd.Grouper(key=key, freq=freq))
        return grouper

    def resample(self, key, freq):  
        # resample somehow mangles the key column so making copy
        index_series = self.series(key)
        self.addSeries('resample_index', index_series)
        # convert time to DateTime format
        self.toDateTime('resample_index') 
        resampled = self.data_frame.resample(freq, on='resample_index').mean()
        self.setDataFrame(resampled)

    def write(self):
        print(self.data_frame)

    def series(self, header):
        series = self.data_frame[header]
        return series

    def addSeries(self, name, series):
        self.data_frame[name] = series

    def describe(self, series):
        retval = series.describe()
        return retval

    def shape(self, data_frame):
        (x,y) = data_frame.shape
        return (x,y)

    def columnData(self, header):
        column = {header: tuple(self.data_frame[header].values)}
        return column

    def normalize(self):
        avg = self.data_frame.mean()
        dev = self.data_frame.std()
        for col in self.data_frame.columns:
            self.data_frame[col] = (self.data_frame[col] - avg.loc[col]) / dev.loc[col]

    def difference(self):
        self.data_frame = self.data_frame.diff().dropna()

    def lagReport(self, series, laggged, maxrange=10):    
        for lag in range(1,maxrange):
            main_series = series.iloc[lag:]
            lagged_series = laggged.iloc[:-lag]
            print('Series: %s'%laggged.name + ' Lag: %s'%lag)
            print(pearsonr(main_series, lagged_series))
            print('------')

    def VARfit(self, data_frame, lags=4):    
        model = VAR(data_frame)
        model.select_order(lags)
        model_fit = model.fit(maxlags=lags)
        return model_fit

    def VARMAXfit(self, endog, exog, order=(2,0), maxiter=1000):
        model = VARMAX(endog.dropna(), order=order, trend='n', exog=exog.dropna())
        model_fit = model.fit(maxiter=maxiter, disp=False)
        return model_fit

    def VARMAXvma(self, endog, order=(2,0), error_cov_type='diagonal', maxiter=1000, disp=False):
        model = VARMAX(endog.dropna(), order=order, error_cov_type=error_cov_type)
        model_fit = model.fit(maxiter=maxiter, disp=disp)
        return model_fit

    def ARIMAfit(self, data_frame, order=(5,1,0)):        
        fits = {}
        for col in data_frame.columns:
            index = data_frame.index
            if (index.name == col):
                continue
            model = ARIMA(data_frame[col].dropna(), order=order)
            model_fit = model.fit(disp=0)
            fits[col] = model_fit
        return fits

    # Generated training sequences for use in the model.
    def createSequences(self, values, time_steps=5):
        output = []
        for i in range(len(values) - time_steps):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)

    def buildAutoencoderModel(self, train, filters=[32,16], kernel_size=7, padding='same', strides=2, activation='relu', rate=0.2, learning_rate=0.001, loss='mse'):
        model = keras.Sequential(
            [
                layers.Input(shape=(train.shape[1], train.shape[2])),
                layers.Conv1D(
                    filters=filters[0], kernel_size=kernel_size, padding=padding, strides=strides, activation=activation
                ),
                layers.Dropout(rate=rate),
                layers.Conv1D(
                    filters=filters[1], kernel_size=kernel_size, padding=padding, strides=strides, activation=activation
                ),
                layers.Conv1DTranspose(
                    filters=filters[1], kernel_size=kernel_size, padding=padding, strides=strides, activation=activation
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=filters[0], kernel_size=kernel_size, padding=padding, strides=strides, activation=activation
                ),
                layers.Conv1DTranspose(filters=1, kernel_size=kernel_size, padding=padding),
            ]
        )

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)
        return model

    def trainAutoencoderModel(self, model, train, epochs=50, batch_size=128, validation_split=0.1, monitor='val_loss', mode='min'):
        history = model.fit(
            train,
            train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor=monitor, patience=5, mode=mode)
            ],
        )
        return history

    def smooth(self, x, window_len=11, window='hanning'):
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w/w.sum(), s, mode='valid')

        return y
