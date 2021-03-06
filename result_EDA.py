import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

faang = ['aapl', 'amzn', 'fb', 'goog', 'nflx']
#dictionaries for storing cnn predictions vs expected 
# and performance evaluations
cnn_pve, cnn_performance_evals = {}, {}

#dictionaries for storing lstm predictions vs expected 
# and performance evaluations
lstm_pve, lstm_performance_evals = {}, {}

#initialize accuracy metric data frames for cnn
cnn_rmse, cnn_mse, cnn_evs, cnn_mae, \
    cnn_msle, cnn_meae, cnn_r_square = [pd.DataFrame() for i in range(0, 7)]

#initialize accuracy metric data frames for lstm
lstm_rmse, lstm_mse, lstm_evs, lstm_mae, \
    lstm_msle, lstm_meae, lstm_r_square = [pd.DataFrame() for i in range(0, 7)]

#all dictionaries to be indexed by faang company ticker symbol
for ticker_sym in faang:
    #read in cnn predictions vs expected
    # and performance evaluations
    cnn_pve[ticker_sym] = pd.read_csv('./Results/cnn/' + ticker_sym + '/pred_vs_exp.csv')
    cnn_performance_evals[ticker_sym] = pd.read_csv('./output/cnn/' + ticker_sym + '/performance_evals.csv')
    #cnn column-wise concatenation of each metric for all companies
    cnn_rmse = pd.concat([cnn_rmse, cnn_performance_evals[ticker_sym]['rmse']], axis=1)
    cnn_mse = pd.concat([cnn_mse, cnn_performance_evals[ticker_sym]['mse']], axis=1)
    cnn_evs = pd.concat([cnn_evs, cnn_performance_evals[ticker_sym]['evs']], axis=1)
    cnn_mae = pd.concat([cnn_mae, cnn_performance_evals[ticker_sym]['mae']], axis=1)
    cnn_msle = pd.concat([cnn_msle, cnn_performance_evals[ticker_sym]['msle']], axis=1)
    cnn_meae = pd.concat([cnn_meae, cnn_performance_evals[ticker_sym]['meae']], axis=1)
    cnn_r_square = pd.concat([cnn_r_square, cnn_performance_evals[ticker_sym]['r_square']], axis=1)
    
    #read in lstm predictions vs expected
    # and performance evaluations
    lstm_pve[ticker_sym] = pd.read_csv('./Results/lstm/' + ticker_sym + '/pred_vs_exp.csv')
    lstm_performance_evals[ticker_sym] = pd.read_csv('./output/lstm/' + ticker_sym + '/performance_evals.csv')
    #cnn column-wise concatenation of each metric for all companies
    lstm_rmse = pd.concat([lstm_rmse, lstm_performance_evals[ticker_sym]['rmse']], axis=1)
    lstm_mse = pd.concat([lstm_mse, lstm_performance_evals[ticker_sym]['mse']], axis=1)
    lstm_evs = pd.concat([lstm_evs, lstm_performance_evals[ticker_sym]['evs']], axis=1)
    lstm_mae = pd.concat([lstm_mae, lstm_performance_evals[ticker_sym]['mae']], axis=1)
    lstm_msle = pd.concat([lstm_msle, lstm_performance_evals[ticker_sym]['msle']], axis=1)
    lstm_meae = pd.concat([lstm_meae, lstm_performance_evals[ticker_sym]['meae']], axis=1)
    lstm_r_square = pd.concat([lstm_r_square, lstm_performance_evals[ticker_sym]['r_square']], axis=1)

#set appropriate column names
cnn_rmse.columns, cnn_mse.columns, cnn_evs.columns, cnn_mae.columns, \
    cnn_msle.columns, cnn_meae.columns, cnn_r_square.columns = [faang for i in range(0, 7)]

lstm_rmse.columns, lstm_mse.columns, lstm_evs.columns, lstm_mae.columns, \
    lstm_msle.columns, lstm_meae.columns, lstm_r_square.columns = [faang for i in range(0, 7)]

#normalize cnn_mse and lstm_msle
cnn_mse = cnn_mse/1000
lstm_msle = lstm_msle * 10

#plot cnn accuracy metrics and save to png
cnn_plotpath = './output/cnn/plots/'
sb.barplot(data=cnn_rmse, ci = None).set(xlabel='Company', ylabel='Root Means Squared Error (CNN)')
plt.savefig(cnn_plotpath + 'cnn_rmse.png')
plt.show()

sb.barplot(data=cnn_mse, ci = None).set(xlabel='Company', ylabel='Means Squared Error (CNN)')
plt.savefig(cnn_plotpath + 'cnn_mse.png')
plt.show()

sb.barplot(data=cnn_evs, ci = None).set(xlabel='Company', ylabel='Explained Variance Score (CNN)')
plt.savefig(cnn_plotpath + 'cnn_evs.png')
plt.show()

sb.barplot(data=cnn_mae, ci = None).set(xlabel='Company', ylabel='Mean Absolute Error (CNN)')
plt.savefig(cnn_plotpath + 'cnn_mae.png')
plt.show()

sb.barplot(data=cnn_msle, ci = None).set(xlabel='Company', ylabel='Mean Squared Log Error (CNN)')
plt.savefig(cnn_plotpath + 'cnn_msle.png')
plt.show()

sb.barplot(data=cnn_meae, ci = None).set(xlabel='Company', ylabel='Median Absolute Error (CNN)')
plt.savefig(cnn_plotpath + 'cnn_meae.png')
plt.show()

sb.barplot(data=cnn_r_square, ci = None).set(xlabel='Company', ylabel='Coefficient of Determination (CNN)')
plt.savefig(cnn_plotpath + 'cnn_r_square.png')
plt.show()


#plot lstm accuracy metrics and save to png
lstm_plotpath = './output/lstm/plots/'
sb.barplot(data=lstm_rmse, ci = None).set(xlabel='Company', ylabel='Root Means Squared Error (LSTM)')
plt.savefig(lstm_plotpath + 'lstm_rmse.png')
plt.show()

sb.barplot(data=lstm_mse, ci = None).set(xlabel='Company', ylabel='Means Squared Error (LSTM)')
plt.savefig(lstm_plotpath + 'lstm_mse.png')
plt.show()

sb.barplot(data=lstm_evs, ci = None).set(xlabel='Company', ylabel='Explained Variance Score (LSTM)')
plt.savefig(lstm_plotpath + 'lstm_evs.png')
plt.show()

sb.barplot(data=lstm_mae, ci = None).set(xlabel='Company', ylabel='Mean Absolute Error (LSTM)')
plt.savefig(lstm_plotpath + 'lstm_mae.png')
plt.show()

sb.barplot(data=lstm_msle, ci = None).set(xlabel='Company', ylabel='Mean Squared Log Error (LSTM)')
plt.savefig(lstm_plotpath + 'lstm_msle.png')
plt.show()

sb.barplot(data=lstm_meae, ci = None).set(xlabel='Company', ylabel='Median Absolute Error (LSTM)')
plt.savefig(lstm_plotpath + 'lstm_meae.png')
plt.show()

sb.barplot(data=cnn_r_square, ci = None).set(xlabel='Company', ylabel='Coefficient of Determination (LSTM)')
plt.savefig(lstm_plotpath + 'lstm_r_square.png')
plt.show()