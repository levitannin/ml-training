#libraries
import pandas as pd
import quandl		#Quandl is now quandl, downloading capitol version will not work.

df = quandl.get('WIKI/GOOGL') #Must be cap to work.
df = df [['Adj. Open', 'Adj.High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]	#indicating which columns are relevant.

#Establishing new values based on assessment of others.
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.00
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

print(df.head())	#printing the current data set.