import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas import DatetimeIndex
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    data_dir = './data'
    datasource = "amzn"
    data = pd.read_csv(os.path.join(data_dir, datasource + '.csv'), header=0, parse_dates={"timestamp": [0]})
    dt_idx = DatetimeIndex(data.timestamp)

    data = data.drop('timestamp', axis=1)

    data = pd.DataFrame(data['High'])

    data.index = dt_idx
    print(data.tail())

    valid_start_dt = '2013-12-06'
    test_start_dt = '2017-01-12'

    data[data.index < valid_start_dt][['High']].rename(columns={'High': 'train'}) \
        .join(data[(data.index >= valid_start_dt) & (data.index < test_start_dt)][['High']] \
              .rename(columns={'High': 'validation'}), how='outer') \
        .join(data[test_start_dt:][['High']].rename(columns={'High': 'test'}), how='outer') \
        .plot(y=['train', 'validation', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('High', fontsize=12)
    #plt.show()

    T = 10
    HORIZON = 1

    train = data.copy()[data.index < valid_start_dt][['High']]

    scaler = MinMaxScaler()
    train['High'] = scaler.fit_transform(train)
    print(train.head(10))

    data[data.index < valid_start_dt][['High']].rename(columns={'High': 'original High'}).plot.hist(bins=100,
                                                                                                        fontsize=12)
    train.rename(columns={'High': 'scaled High'}).plot.hist(bins=100, fontsize=12)
    #plt.show()

    train_shifted = train.copy()
    train_shifted['y_t+1'] = train_shifted['High'].shift(-1, freq='D')
    print(train_shifted.head(10))

    for t in range(1, T + 1):
        train_shifted['High_t-' + str(T - t)] = train_shifted['High'].shift(T - t, freq='D')
    train_shifted = train_shifted.rename(columns={'High': 'High_original'})
    print(train_shifted.head(10))

    #train_shifted = train_shifted.dropna(how='any')

    print(train_shifted)

    y_train = train_shifted[['y_t+1']].as_matrix()

    print(y_train.shape)

    print(y_train[:3])

    X_train = train_shifted[['High_t-' + str(T - t) for t in range(1, T + 1)]].as_matrix()
    X_train = X_train[..., np.newaxis]

    print(X_train.shape)

    print(X_train[:3])

    look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d') - dt.timedelta(days=T-1)

    valid = data.copy()[(data.index >= look_back_dt) & (data.index < test_start_dt)][['High']]
    print(valid.head())

    valid['High'] = scaler.transform(valid)
    print(valid.head())

    valid_shifted = valid.copy()
    valid_shifted['y+1'] = valid_shifted['High'].shift(-1, freq='D')
    for t in range(1, T + 1):
        valid_shifted['High_t-' + str(T - t)] = valid_shifted['High'].shift(T - t, freq='D')
    #valid_shifted = valid_shifted.dropna(how='any')
    y_valid = valid_shifted['y+1'].as_matrix()
    X_valid = valid_shifted[['High_t-' + str(T - t) for t in range(1, T + 1)]].as_matrix()
    X_valid = X_valid[..., np.newaxis]

    print(y_valid.shape)
    print(X_valid.shape)