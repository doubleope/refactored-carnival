import pandas as pd

faang = ['aapl', 'amzn', 'fb', 'goog', 'nflx']
#dictionaries for storing cnn predictions vs expected 
# and performance evaluations
cnn_pve, cnn_performance_evals = {}, {}

#dictionaries for storing lstm predictions vs expected 
# and performance evaluations
lstm_pve, lstm_performance_evals = {}, {}

#all dictionaries to be indexed by faang company ticker symbol
for ticker_sym in faang:
    #reading in cnn predictions vs expected
    # and performance evaluations
    cnn_pve[ticker_sym] = pd.read_csv('./Results/cnn/' + ticker_sym + '/pred_vs_exp.csv')
    cnn_performance_evals[ticker_sym] = pd.read_csv('./output/cnn/' + ticker_sym + '/performance_evals.csv')

    #reading in lstm predictions vs expected
    # and performance evaluations
    lstm_pve[ticker_sym] = pd.read_csv('./Results/lstm/' + ticker_sym + '/pred_vs_exp.csv')
    lstm_performance_evals[ticker_sym] = pd.read_csv('./output/lstm/' + ticker_sym + '/performance_evals.csv')

