import pandas as pd
import pandas_datareader as dr
import matplotlib

companies = ["AAPL", "AMZN", "UBER", "NOK", "F", "GE", "FB", "NFLX", "GOOG",
            "PINS", "AMD", "MSFT", "IBM", "INTC", "BAC", "SNAP", "TWTR", "T"]

for i in companies:
    df = dr.get_data_yahoo(i, start = '2010-02-08', end = '2020-02-08')
    df.to_csv("../data/"+ i +"_data.csv")

