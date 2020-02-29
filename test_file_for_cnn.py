import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import DatetimeIndex
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
    plt.show()

    T = 10
    HORIZON = 1

    train = data.copy()[data.index < valid_start_dt][['High']]

    scaler = MinMaxScaler()
    train['High'] = scaler.fit_transform(train)
    train.head(10)

    data[data.index < valid_start_dt][['High']].rename(columns={'High': 'original High'}).plot.hist(bins=100,
                                                                                                        fontsize=12)
    train.rename(columns={'High': 'scaled High'}).plot.hist(bins=100, fontsize=12)
    plt.show()