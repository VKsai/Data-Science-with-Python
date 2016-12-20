#Data Preprocessing Using sklearn libraries
#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#READING DATA
#Type your data file name replacing <data file name>
#In this case it is Comma Sepearated Values, it can be other files like 
#excel file, et cetera.     
data = pd.read_csv('<data file name>');
X = data.iloc[:, :-1].values #: is for all elements and -1 is for removing last column
Y = data.iloc[:, 3].values # [:, :] == [rows, columns]

#using sklearn's imputer to cement up the MISSING VALUES
#Defined in line 6
#The default strategy for filling up the missing value is to fill NAN's with
#respective column's mean.
#use sklearn's Imputer class with the respective parameters
#first define missing value type; usually missing_values = 'NaN', if the array has "nan"
#for blank spaces.
#second define the strategy going to be used. In this case the strategy is to replace the 
# "nan" or blank spaces with respective means of the column.
i = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#FITTING or INSERTING UPDATED VALUES TO DESIRED MATRIX
#Now that the Imputer is called we need to fit the means into their respective positions
#in the data set.
#The X matrix with the columns which needs to be transformed are passed. 
iputer_fit = i.fit(X[:, 1:3])

#TRANSFORMING THE MATRIX
#Applying Imputer's transform method with X as parameter and hence updating X.
X[:, 1:3] = i.transform(X[:, 1:3])

#BRINGING LABELS DOWN TO NUMBERS
#Once the missing values or nan's are replaced using a prticular strategy, we can then 
#work on categorizing data to some reconizable format
#Let us define the required library at line 7
labelEncoder_variable = LabelEncoder()

#FITTING NUMBERED DATA
#Now that we have initiated the object of LabelEncoder in line 39
#Next we need to fit that into the matrix X
X[:, 0] = labelEncoder_variable.fit_transform(X[:, 0])

#Next use of ONEHOTENCODER which will be required in later sections where we will 
#try to implement SVM and Linear Models such as Linear Regression, Logistic Regression, etc.
#OneHotEncoder class is imported in line 8.
oneHotEncoder_object = OneHotEncoder(categorical_features = [0])

#Now we need to fit the object back to our Matrix X containing the Dataset
#Converting the dataset to array is useful because we can access data by indices
X = oneHotEncoder_object.fit_transform(X).toarray()
