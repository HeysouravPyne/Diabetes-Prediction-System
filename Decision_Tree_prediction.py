# Importing the libraries
import numpy as np
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15, random_state= 42)


#Data standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Fitting Decision Tree Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
# 'gini' or 'entropy'
classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 0)
#classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)

classifier.fit(x_train, y_train)
y_pred_train = classifier.predict(x_train)
y_pred_test = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of the training data :: ",accuracy_score(y_train, y_pred_train))
print("Accuracy score of the testing data :: ",accuracy_score(y_test, y_pred_test))








# CUSTOM INPUT
input_data = (5,166,72,19,175,25.8,0.578,51)

# Changing the input data to an numpy array
input_data = np.asarray(input_data)

# Reshape the data as we are predicting for one instance
input_data = input_data.reshape(1,-1)

# Standardize the input data
input_data = sc.transform(input_data)

prediction = classifier.predict(input_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic.')
else:
    print('The person is diabetic.')










