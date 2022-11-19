# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 400, criterion = 'gini', random_state = 42)

classifier.fit(x_train,y_train)
y_pred_train = classifier.predict(x_train)
y_pred_test = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of the training data :: ",accuracy_score(y_train, y_pred_train))
print("Accuracy score of the testing data :: ",accuracy_score(y_test, y_pred_test))