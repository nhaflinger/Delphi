import os 
import glob
import re
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import delphi.core as dp
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


def main():
# parse command line arguments
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-d", "--directory", default='', help="input directory")
    parser.add_argument("-f", "--filename", default='', help="input file")
    parser.add_argument("-p", "--pattern", default='', help="glob style pattern")
    parser.add_argument("-x", "--regex", default='', help="regular expression")
    args = parser.parse_args()

    directory = args.directory
    filename = args.filename
    pattern = args.pattern
    regex = args.regex
    fullpath = directory + '\/' + filename
    
    # create Delphi object
    c  = dp.core()

    # get all files in directory
    #if (pattern != ''):
        #file_names = c.getFileList(directory, pattern)
    #for file_name in file_names:
        #print(file_name)

    file_exists = os.path.exists(fullpath)
    if (file_exists == False):
        print('File does not exist"')
        return

    print("Loading data for " + fullpath)
    df = c.readCSV(fullpath)
    if (df.empty):
        print('Empty data frame returned')
        return
    
    # resample 
    #c.resample('gfms_abs_time', '20L')

    # convert time to DateTime format
    c.toDateTime('gfms_abs_time')

    gfms_abs_time = c.series('gfms_abs_time')

    keys = ['h', 'lox', 'ipa', 'srv', 'purge', 'bat_a', 'bat_b', 'ebus', 'hinge_x_amps[EXP]', 'hinge_y_amps[EXP]', 'v_ipa_amps[EXP]', 
            'v_lox_amps[EXP]', 'p_ipa[EXP]', 'p_lox[EXP]', 'p_c[EXP]', 'p_c2[EXP]', 't.ofms_temp', 't.sensor_current', 't.vehicle_solenoid_current', 
            't.acs_stack_current', 't.bus_b_current', 't.ofms_stack_current']

    # normalize and take first difference to remove trend
    p_ipa = c.series('p_ipa[EXP]')
    p_lox = c.series('p_lox[EXP]')
    p_c = c.series('p_c[EXP]')
    frame = { 'p_ipa': p_ipa, 'p_lox': p_lox, 'p_ipa': p_ipa, 'p_lox': p_lox, 'p_c': p_c }
    df2 = pd.DataFrame(frame)
    c.setDataFrame(df2)

    # normalize the data
    c.normalize()

    # take first difference of data
    c.difference()

    # nothing equivalent to "volatility" to remove (std deviation trends)
    #df2.index = pd.DatetimeIndex(gfms_abs_time)
    #volatility = df2.groupby(df2.index.second).std()

    # nothing equivalent to "seasonality" to remove (repeating cycles)
    
    # determine lags
    p_ipa_diff = c.series('p_ipa')
    p_lox_diff = c.series('p_lox')
    p_c_diff = c.series('p_c')
    #plot_pacf(p_ipa_diff)
    #plot_pacf(p_lox_diff)
    #plot_pacf(p_c_diff)

    # ARIMA test
    frame = { 'gfms_abs_time': gfms_abs_time, 'p_ipa_diff': p_ipa_diff, 'p_lox_diff': p_lox_diff}
    df2 = pd.DataFrame(frame)
    df2.index = pd.DatetimeIndex(gfms_abs_time)
    df2.index = pd.DatetimeIndex(df2.index).to_period('20L') 
    model_fits = c.ARIMAfit(df2)
    c.ARIMAresidualsPlot(df2, model_fits)
    plt.show()
    
    # plot correlations
    frame = { 'p_ipa_diff': p_ipa_diff, 'p_lox_diff': p_lox_diff, 'p_c_diff': p_c_diff }
    df2 = pd.DataFrame(frame)
    c.autocorrelationPlot(df2)

    frame = { 'gfms_abs_time': gfms_abs_time, 'p_ipa_diff': p_ipa_diff, 'p_lox_diff': p_lox_diff }
    df2 = pd.DataFrame(frame)
    c.setDataFrame(df2)
    
    # look at range of lag values
    #c.lagReport(p_ipa_diff, p_lox_diff, 10)
    #print('\n\n')
    #c.lagReport(p_ipa_diff, p_c_diff, 10)

    df2.index = pd.DatetimeIndex(gfms_abs_time)
    df2.index = pd.DatetimeIndex(df2.index).to_period('20U') 
    df2 = df2[['p_ipa_diff', 'p_lox_diff']].dropna()
    #print(df2.head(10)) 
    #df2['p_ipa_diff'].plot(title = 'p_ipa_diff')
    
    # vector auto-regession test
    #lox_model_fit = c.VARfit(df2, lags=4)
    #print(lox_model_fit.summary())
    #lox_model_fit.plot_acorr()

    #lag_order = lox_model_fit.k_ar
    #lox_model_fit.forecast(df2.values[-lag_order:], 5)
    #lox_model_fit.plot_forecast(10)

    #yval_list = ['p_ipa_diff', 'p_lox_diff', 'p_c_diff']

    #fig = c.simplePlot("Normalized p_ipa", 'gfms_abs_time', 'norm_p_ipa')

    #fig = c.dualPlot("Velocity", 'gfms_abs_time', 'gps_vel_ecef[1]', 'gps_vel_ecef[2]')
    #yval_list = ['p_ipa[EXP]', 'p_lox[EXP]', 'p_c[EXP]']
    #if (regex == ''):
        #print('No regular expression')
    #else:
        #yval_list = c.seriesSearch(regex)

    #yval_list = ['p_ipa_norm', 'p_lox_norm']
    #fig = c.multiPlot("Normalized values", 'gfms_abs_time', yval_list)
    
    yval_list = ['p_ipa_diff', 'p_lox_diff']
    #fig = c.multiPlot("First difference", 'gfms_abs_time', yval_list)

    #fig = c.stackedPlot("GPS Velocity", 'gfms_abs_time', yval_list)

    plt.show()

if __name__ == "__main__":
    main()