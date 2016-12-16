#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

#reading data
#Type your data file name replacing <data file name>
#In this case it is Comma Sepearated Values, it can be other files like 
#excel file, et cetera.
data = pd.read_csv('<data file name>');
X = data.iloc[:, :-1].values #: is for all elements and -1 is for removing last column
Y = data.iloc[:, 3].values # [:, :] == [rows, columns]

#using sklearn's imputer to cement up the missing values
#The default strategy for filling up the missing value is to fill NAN's with
#respective column's mean.
#use sklearn's Imputer class with the respective parameters
#first define missing value type; usually missing_values = 'NaN', if the array has "nan"
#for blank spaces.
#second define the strategy going to be used. In this case the strategy is to replace the 
# "nan" or blank spaces with respective means of the column.
i = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#Now that the Imputer is called we need to fit the means into their respective positions
#in the data set.
#The X matrix with the columns which needs to be transformed are passed. 
iputer_fit = i.fit(X[:, 1:3])

#Applying Imputer's transform method with X as parameter and hence updating X.
X[:, 1:3] = i.transform(X[:, 1:3])

#Printing X you can see how the spaces are replaced by respective means.
print(X)
