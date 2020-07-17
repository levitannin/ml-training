#Libraries
import pandas as pd
import quandl 			#Now quandl, not Quandl
import math

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

#Establishing corrolation in order to make processing easier.
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)	#Cannot work with N/A with pandas.  Must use this method to create an outlire.

forecast_out = int(math.ceil(0.01 * len(df)))

#Creation of a new variable, shifting the old data with the new data variable.
df['label'] = df[forecast_col].shift(- forecast_out)
df.dropna(inplace = True)

print(df.head())
