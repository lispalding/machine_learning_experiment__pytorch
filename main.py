# MADE BY: Lisette Spalding
# FILE NAME: main.py
# PROJECT NAME: machine_learning_experiment__pytorch
# DATE CREATED: 04/29/2021
# DATE LAST MODIFIED: 04/29/2021
# PYTHON VER. USED: 3.x

############################## IMPORTS ##############################
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib import style
import pickle

from os import path
################################ FIN ################################

############################ FOLDER SETUP ############################
generalFolder = path.dirname(__file__) # General folder set-up
dataFolder = path.join(generalFolder, "data_sets")
dataSet = path.join(dataFolder, "student-mat.csv")
################################ FIN ################################

############################# DATA SETUP #############################
## Collecting and loading data:
data = pd.read_csv(dataSet, sep = ";") # Since our data is separated with semicolons we need to use this: sep = ";"
print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # Collecting only the relevant data from the data set
## Data collection and loading FIN

## Separating the data:
predict = "G3"

x = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels
## Separating the data FIN

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
################################ FIN ################################

############################# ALGORITHM #############################
linear = linear_model.LinearRegression() # Defining the linear model that will be used

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print(accuracy) # Checking how well the algorithm preformed on the test

## Viewing Constants:
print("Coefficient: \n", linear.coef_) # These are each slope value
print("Intercept: \n", linear.intercept_) # This is the intercept
## Constants FIN

predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
################################ FIN ################################
