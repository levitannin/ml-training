#Regression Training and Testing - Practical Machine Learning Tutorial with Python p.4
#sklearn has updated since 2016; Cross_Validation is no longer a sublibrary.
#used sklearn.model_selection instead to use train_test_split

import pandas as pd
import quandl, math
import numpy as np 
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)	#Cannot work with N/A with pandas.  Must use this method to create an outlire.

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(- forecast_out)
df.dropna(inplace = True)

#defining variables
X = np.array(df.drop(['label'], 1))
#dropping label column; returns new df which is being converted into np.array and stored in variable.
y = np.array(df['label'])

#Processing and storing back in the variable.  
X = preprocessing.scale(X)

#redefining
#X = X[:-forecast_out + 1] doesn't work because of dropna above.
#df.dropna(inplace = True)
y = np.array(df['label'])

#This will take 20% of the data and use it to test the cross validation, using the other 80% to train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n-jobs = 10)   #How many jobs you want run in parallel.
clf.fit(X_train, y_train)					#syn with train
accuracy = clf.score(X_test, y_test)		#syn with test


print(accuracy)
