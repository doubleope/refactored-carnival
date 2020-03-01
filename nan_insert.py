import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex


#This script only works if run from the data folder

target = pd.read_csv("./data/amzn.csv")
#target.Date is read in as 'str' type
#converting to datetime
target.Date = pd.to_datetime(target.Date)
colnames = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
dt_idx = DatetimeIndex(freq='D', start='2004-08-19', end='2020-02-21')
#setting up df as a data frame with datetime index
#under column specified as 'Date'
df = pd.DataFrame(dt_idx, columns=['Date'])

#setting nan's
for i in colnames:
    df[i] = np.nan

#keep in mind that index has not been set
#for either target or df

#concatenating df observations to target
target_full_noindex = pd.concat([target, df], axis=0, join='outer', sort=True)

#produces array where the position of the second duplicate is marked as True
#while that of the first is marked as false, only searches for duplicates based on 'Date'
dupes = target_full_noindex.duplicated(['Date'], keep='first')

#subsetting only observations marked as False (marked as non-duplicates)
#P.S. the symbol '~' means negation for pandas
target_full_noindex = target_full_noindex[~dupes]

#sorting based on date
target_full_noindex = target_full_noindex.sort_values('Date')

#setting index to 'Date' column
target_full_noindex = target_full_noindex.set_index('Date')

print(target_full_noindex.head(10))