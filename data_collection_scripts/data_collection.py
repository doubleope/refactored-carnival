import pandas as pd
import pandas_datareader as dr
import matplotlib


#this is for when I tried to use alpha vantage, yahoo is better in the end
#available hourly data is insufficient, using daily instead
#do not delete below comment in case we need it again
# ts = TimeSeries(key=AV_key, output_format='pandas')


FAANG = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOG']
# companies = ["AAPL", "AMZN", "UBER", "NOK", "F", "GE", "FB", "NFLX", "GOOG",
#             "PINS", "AMD", "MSFT", "IBM", "INTC", "BAC", "SNAP", "TWTR", "T"]

for company_name in FAANG:
    #start date might say 1950, only goes as far as data is available
    df = dr.get_data_yahoo(company_name, start = '1950-02-08', end = '2020-02-08')
    df.to_csv("./data/"+ company_name +".csv")
