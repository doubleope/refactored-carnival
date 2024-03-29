from matplotlib import pyplot
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, mean_squared_log_error, \
    median_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy
from modify_data import load_modified_data
import time
from matplotlib import pyplot

#for command line argument and saving to file
import sys
from os import path

prog_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

<<<<<<< HEAD

series = load_modified_data("amzn")
=======
#uses command line argument
series = load_modified_data(sys.argv[1])
#amzn for testing
# series = load_modified_data("amzn")
#for testing with rows 0-10
# series = series[0:10]
>>>>>>> 49844620b8126f87900addc9ce85b82ed59902e1
series = series.squeeze()

raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets (70% for training and 30% for testing)
training_limit = int(0.7 * len(supervised_values))
train, test = supervised_values[0: training_limit - 1], supervised_values[
                                                        -((len(supervised_values) - training_limit) + 1):]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
pred_vs_exp = DataFrame(columns=['predictions', 'actual'])
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Predicted=%f, Expected=%f' % (yhat, expected))

actual = raw_values[-((len(supervised_values) - training_limit) + 1):]

# test to see if accuracy is better if nan values are replaced with 0
# convert actual and predictions to dataframe and remove rows where actual is nan
df = DataFrame({'actual': actual, 'predictions': predictions})
df = df[df['actual'].notna()]
df = df[df['predictions'].notna()]
actual = numpy.asarray(df.actual)
predictions = list(df.predictions)

#save prediction vs expected to data frame and csv
#TODO: change path to ./Results/lstm/.../pred_vs_exp.csv
df.to_csv("./Results/lstm/" + sys.argv[1] + "/pred_vs_exp.csv", index=False)




# evaluate performance
rmse = sqrt(mean_squared_error(actual, predictions))
mse = mean_squared_error(actual, predictions)
evs = explained_variance_score(actual, predictions)
mae = mean_absolute_error(actual, predictions)
msle = mean_squared_log_error(actual, predictions)
meae = median_absolute_error(actual, predictions)
r_square = r2_score(actual, predictions)
print("rmse: ", rmse, " mse: ", mse, "evs: ", evs, "mae: ", mae, "msle: ", msle, "meae: ", meae, "r_square: ", r_square)

#save accuracy metrics to data frame
performance_evals = DataFrame(columns=['rmse', 'mse', 'evs', 'mae', \
                                        'msle', 'meae', 'r_square'])
performance_evals = performance_evals.append({'rmse':rmse, \
                                                'mse':mse, \
                                                'evs':evs, \
                                                'mae':mae, \
                                                'msle':msle, \
                                                'meae':meae, \
                                                'r_square': r_square}, \
                                                ignore_index=True)

#check if respective files already exist
pe_path = './output/lstm/' + sys.argv[1] + '/performance_evals.csv'
#for testing with amzn
# pe_path = './output/lstm/' + 'amzn' + '/performance_evals.csv'
if path.exists(pe_path):
    stored_pe = read_csv(pe_path)
    performance_evals = concat([stored_pe, performance_evals])
#if exists, appends, if not creates and writes to file
performance_evals.to_csv(pe_path, index=False)

# save scores to file
scores = open("./output/scores.txt", "a+")
scores.write("rmse: " + str(rmse) + " mse: " + str(mse) + " evs: " + str(evs) + " mae: " + str(mae) + " msle: " + \
             str(msle) + " meae: " + str(meae) + " r_square: " + str(r_square))
scores.close()

# save runtime to file
prog_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
f = open("./output/amzn/runtime.txt", "a+")
time_output = "\nProgram started at: " + prog_start + " and ended at: " + prog_end
f.write(time_output)
f.close()

df.plot(style=['r', 'b'])

pyplot.savefig("./output/lstm/amzn/plot.png")
