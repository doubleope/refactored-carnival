from glob import glob
from math import sqrt
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, mean_squared_log_error, \
    median_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import pandas as pd
import datetime as dt
from modify_data import load_modified_data

from common.utils import load_data, mape

if __name__ == '__main__':
    data = load_modified_data("amzn")

    valid_start_dt = '2013-12-06'
    test_start_dt = '2017-01-12'

    # data[data.index < valid_start_dt][['High']].rename(columns={'High': 'train'}) \
    #     .join(data[(data.index >= valid_start_dt) & (data.index < test_start_dt)][['High']] \
    #           .rename(columns={'High': 'validation'}), how='outer') \
    #     .join(data[test_start_dt:][['High']].rename(columns={'High': 'test'}), how='outer') \
    #     .plot(y=['train', 'validation', 'test'], figsize=(15, 8), fontsize=12)
    # plt.xlabel('timestamp', fontsize=12)
    # plt.ylabel('High', fontsize=12)
    # plt.show()

    T = 10
    HORIZON = 1

    train = data.copy()[data.index < valid_start_dt][['High']]

    scaler = MinMaxScaler()
    train['High'] = scaler.fit_transform(train)
    print(train.head(10))

    # data[data.index < valid_start_dt][['High']].rename(columns={'High': 'original High'}).plot.hist(bins=100,
    #                                                                                                     fontsize=12)
    # train.rename(columns={'High': 'scaled High'}).plot.hist(bins=100, fontsize=12)
    # plt.show()

    train_shifted = train.copy()
    train_shifted['y_t+1'] = train_shifted['High'].shift(-1, freq='D')
    print(train_shifted.head(10))

    for t in range(1, T + 1):
        train_shifted['High_t-' + str(T - t)] = train_shifted['High'].shift(T - t, freq='D')
    train_shifted = train_shifted.rename(columns={'High': 'High_original'})
    print(train_shifted.head(10))

    train_shifted = train_shifted.dropna(how='any')

    print(train_shifted)

    y_train = train_shifted[['y_t+1']].as_matrix()

    print(y_train.shape)

    print(y_train[:3])

    X_train = train_shifted[['High_t-' + str(T - t) for t in range(1, T + 1)]].as_matrix()
    X_train = X_train[..., np.newaxis]

    print(X_train.shape)

    print(X_train[:3])

    look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d') - dt.timedelta(days=T - 1)

    valid = data.copy()[(data.index >= look_back_dt) & (data.index < test_start_dt)][['High']]
    print(valid.head())

    valid['High'] = scaler.transform(valid)
    print(valid.head())

    valid_shifted = valid.copy()
    valid_shifted['y+1'] = valid_shifted['High'].shift(-1, freq='D')
    for t in range(1, T + 1):
        valid_shifted['High_t-' + str(T - t)] = valid_shifted['High'].shift(T - t, freq='D')
    valid_shifted = valid_shifted.dropna(how='any')
    y_valid = valid_shifted['y+1'].as_matrix()
    X_valid = valid_shifted[['High_t-' + str(T - t) for t in range(1, T + 1)]].as_matrix()
    X_valid = X_valid[..., np.newaxis]

    print(y_valid.shape)
    print(X_valid.shape)

    LATENT_DIM = 5
    KERNEL_SIZE = 2
    BATCH_SIZE = 32
    EPOCHS = 10

    model = Sequential()
    model.add(
        Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=1,
               input_shape=(T, 1)))
    model.add(
        Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=2))
    model.add(
        Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=4))
    model.add(Flatten())
    model.add(Dense(HORIZON, activation='linear'))

    print(model.summary())

    model.compile(optimizer='Adam', loss='mse')

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

    best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)

    history = model.fit(X_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_valid, y_valid),
                        callbacks=[earlystop, best_val],
                        verbose=1)

    best_epoch = np.argmin(np.array(history.history['val_loss'])) + 1
    model.load_weights("model_{:02d}.h5".format(best_epoch))

    # plot_df = pd.DataFrame.from_dict({'train_loss': history.history['loss'], 'val_loss': history.history['val_loss']})
    # plot_df.plot(logy=True, figsize=(10, 10), fontsize=12)
    # plt.xlabel('epoch', fontsize=12)
    # plt.ylabel('loss', fontsize=12)
    # plt.show()

    look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d') - dt.timedelta(days=T - 1)
    test = data.copy()[test_start_dt:][['High']]
    print(test.head())

    test['High'] = scaler.transform(test)
    print(test.head())

    test_shifted = test.copy()
    test_shifted['y_t+1'] = test_shifted['High'].shift(-1, freq='D')
    for t in range(1, T + 1):
        test_shifted['High_t-' + str(T - t)] = test_shifted['High'].shift(T - t, freq='D')
    test_shifted = test_shifted.dropna(how='any')
    y_test = test_shifted['y_t+1'].as_matrix()
    X_test = test_shifted[['High_t-' + str(T - t) for t in range(1, T + 1)]].as_matrix()
    X_test = X_test[..., np.newaxis]

    predictions = model.predict(X_test)
    print(predictions)

    eval_df = pd.DataFrame(predictions, columns=['t+' + str(t) for t in range(1, HORIZON + 1)])
    eval_df['timestamp'] = test_shifted.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(y_test).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    print(eval_df.head())
    actual = eval_df['actual']
    predictions = eval_df['prediction']

    # print(mape(eval_df['prediction'], eval_df['actual']))

    # eval_df[eval_df.timestamp < '2017-01-30'].plot(x='timestamp', y=['prediction', 'actual'], style=['r', 'b'],
    #                                                figsize=(15, 8))
    # plt.xlabel('timestamp', fontsize=12)
    # plt.ylabel('High', fontsize=12)
    # plt.show()

    # remove model files
    for m in glob('model_*.h5'):
        os.remove(m)

    rmse = sqrt(mean_squared_error(actual, predictions))
    mse = mean_squared_error(actual, predictions)
    evs = explained_variance_score(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    msle = mean_squared_log_error(actual, predictions)
    meae = median_absolute_error(actual, predictions)
    r_square = r2_score(actual, predictions)
    print("rmse: ", rmse, " mse: ", mse, "evs: ", evs, "mae: ", mae, "msle: ", msle, "meae: ", meae, "r_square: ",
          r_square)
