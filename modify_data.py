import numpy as np
import pandas as pd
import os
from pandas.core.indexes.datetimes import DatetimeIndex


def load_modified_data(data_source):
    data_dir = './data'

    target = pd.read_csv(os.path.join(data_dir, data_source + '.csv'), header=0, parse_dates={"timestamp": [0]})
    # target.Date is read in as 'str' type
    # converting to datetime
    target.Date = pd.to_datetime(target.timestamp)
    colnames = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
    dt_idx = DatetimeIndex(freq='D', start='2004-08-19', end='2020-02-21')
    # setting up df as a data frame with datetime index
    # under column specified as 'Date'
    df = pd.DataFrame(dt_idx, columns=['timestamp'])

    # setting nan's
    for i in colnames:
        df[i] = np.nan

    # keep in mind that index has not been set
    # for either target or df

    # concatenating df observations to target
    target_full_noindex = pd.concat([target, df], axis=0, join='outer', sort=True)

    # produces array where the position of the second duplicate is marked as True
    # while that of the first is marked as false, only searches for duplicates based on 'timestamp'
    dupes = target_full_noindex.duplicated(['timestamp'], keep='first')

    # subsetting only observations marked as False (marked as non-duplicates)
    # P.S. the symbol '~' means negation for pandas
    target_full_noindex = target_full_noindex[~dupes]
    # sorting based on timestamp
    target_full_noindex = target_full_noindex.sort_values('timestamp')

    # replace nan values with values of each preceding row
    i = 1
    while i != len(target_full_noindex.index):
        if np.isnan(target_full_noindex.iloc[i].High):
            target_full_noindex.loc[i, "High"] = target_full_noindex.iloc[i - 1].High
        i += 1
    # make timestamp the index column
    target_full_noindex = target_full_noindex.set_index('timestamp')

    # get only the 'High' column
    target_full_noindex = pd.DataFrame(target_full_noindex["High"])
    return target_full_noindex


# print(load_modified_data("amzn").head())

