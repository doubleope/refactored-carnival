import numpy as np
import pandas as pd
import os


def load_modified_data(data_source):
    data_dir = './data'

    data = pd.read_csv(os.path.join(data_dir, data_source + '.csv'), header=0, parse_dates={"timestamp": [0]})

    colnames = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
    dt_idx = pd.date_range(freq='D', start='2004-08-19', end='2020-02-21')
    # setting up df as a data frame with datetime index
    # under column specified as 'Date'
    df = pd.DataFrame(dt_idx, columns=['timestamp'])

    # setting nan's
    for i in colnames:
        df[i] = np.nan

    # keep in mind that index has not been set
    # for either target or df

    # concatenating df observations to target
    modified_data = pd.concat([data, df], axis=0, join='outer', sort=True)

    # produces array where the position of the second duplicate is marked as True
    # while that of the first is marked as false, only searches for duplicates based on 'timestamp'
    dupes = modified_data.duplicated(['timestamp'], keep='first')

    # subsetting only observations marked as False (marked as non-duplicates)
    # P.S. the symbol '~' means negation for pandas
    modified_data = modified_data[~dupes]

    # sorting based on timestamp
    modified_data = modified_data.sort_values('timestamp')

    modified_data = modified_data.reset_index()

    # replace nan values with values of each preceding row
    i = 1
    while i != len(modified_data.index):
        if np.isnan(modified_data.iloc[i].High):
            modified_data.loc[i, "High"] = modified_data.iloc[i - 1].High
        i += 1

    # get only the timestamp and High column
    modified_data = modified_data[['timestamp', 'High']].copy()

    # setting index to 'timestamp' column
    modified_data = modified_data.set_index('timestamp')
    return modified_data

print(load_modified_data("amzn").head())