#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:11:38 2020

@author: andreastsoumpariotis
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.api import add_constant
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.metrics import confusion_matrix
import statistics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from IPython.display import display, HTML

# Question 10

pd.set_option('display.max_columns', None)

# Part a
# Summary
weekly = pd.read_csv('weekly.csv')
weekly.describe(include='all')

Year = weekly['Year']
Lag1 = weekly['Lag1']
Lag2 = weekly['Lag2']
Lag3 = weekly['Lag3']
Lag4 = weekly['Lag4']
Lag5 = weekly['Lag5']
Volume = weekly['Volume']
Today = weekly['Today']
Direction = weekly['Direction']

# Correlation
corr = weekly.corr()
print(corr)
# Scatterplot
plt.scatter(weekly['Year'], weekly['Volume'])
plt.title('Volume vs. Index')
plt.ylabel('Volume')
plt.xlabel('Index')

# Part b
predictor = weekly.columns[1:7]
x = sm.add_constant(weekly[predictor])
y = np.array([1 if el=='Up' else 0 for el in weekly.Direction.values])
logit = sm.Logit(y, x)
result = logit.fit()
print(result.summary()))

# Part c
y_predicted = result.predict(x)
y_predicted = np.array(y_predicted > 0.7, dtype=float)
# Confusion Matrix
matrix = np.histogram2d(y_predicted, y, bins=2)[0]
print(pd.DataFrame(matrix, ['Down', 'Up'], ['Down', 'Up']))

# Perc of Correct Predictions
n=1089
((matrix[1,-1] + matrix[0,0])/n)*100 #56.10651974288338
# Training Error Rate
100 - ((matrix[1,-1] + matrix[0,0])/n)*100 #43.89348025711662
# Up
(matrix[1,-1]/(matrix[0,1] + matrix[1,-1]))*100 #92.06611570247934
# Down
(matrix[0,0]/(matrix[1,0] + matrix[0,0]))*100 #11.15702479338843

# Part d

# Splitting the data
# Training Data (2008 and before)
x_train = sm.add_constant(weekly[weekly.Year <= 2008].Lag2)
response_train = weekly[weekly.Year <= 2008].Direction
y_train = np.array([1 if el=='Up' else 0 for el in response_train])
# Testing Data (2009 and 2010)
x_test = sm.add_constant(weekly[weekly.Year > 2008].Lag2)
response_test = weekly[weekly.Year > 2008].Direction
y_test = np.array([1 if el=='Up' else 0 for el in response_test])
# Classifier and Fit
logit = sm.Logit(y_train, x_train)
results = logit.fit()
print(results.summary())
# Predicting the Test Responses
y_predicted2 = results.predict(x_test)
y_predicted2 = np.array(y_predicted2 > 0.5, float)
# Confusion Matrix
matrix2 = np.histogram2d(y_predicted2, y_test,2)[0]
print(pd.DataFrame(matrix2, ['Down', 'Up'], ['Down', 'Up']))

# Perc of Correct Predictions
n2 = 104
((matrix2[1,-1] + matrix2[0,0])/n2)*100 #62.5
# Training Error Rate
100 - ((matrix2[1,-1] + matrix2[0,0])/n2)*100 #37.5
# Up
(matrix2[1,-1]/(matrix2[0,1] + matrix2[1,-1]))*100 #91.80327868852459
# Down
(matrix2[0,0]/(matrix2[1,0] + matrix2[0,0]))*100 #20.930232558139537

# Part e

# LDA
lda = LDA('lsqr', True)

# Splitting the data
# Training Data (2008 and before)
x_training = weekly[weekly.Year <= 2008].Lag2.values
x_training = x_training.reshape((len(x_training),1))
y_training = np.array([1 if el=='Up' else 0 for el in response_train])

# Testing Data (2009 and 2010)
x_test = weekly[weekly.Year > 2008].Lag2.values
x_test = x_test.reshape((len(x_test),1))
y_test = np.array([1 if el=='Up' else 0 for el in response_test])

# Fitting
linear.fit(x_training, y_training)
# Predicting the Test Responses
y_predicted3 = lda.predict(x_test)
y_predicted3 = np.array(y_predicted3 > 0.5, float)

# Confusion Matrix
matrix3 = np.histogram2d(y_predicted3, y_test,2)[0]
print(pd.DataFrame(matrix3, ['Down', 'Up'], ['Down', 'Up']))

# Part f

# QDA

q = QuadraticDiscriminantAnalysis()
q.fit(x_training,y_training)

# Predict Test Data and Evaluate
y_predict = q.predict(x_test)
y_predicted= np.array(y_predict > 0.5, dtype=float)

# Confusion Matrix
matrix4 = np.histogram2d(y_predict, y_test, bins=2)[0]
print(pd.DataFrame(matrix3, ['Down', 'Up'], ['Down', 'Up']))
# Test Error Rate
print((100*np.mean(y_predicted != y_test))) #41.34615384615385

# Part g
# KNN
KNN = KNeighborsClassifier(n_neighbors=1)
KNN.fit(x_training, y_training)

# Test Data
y_predicted = KNN.predict(x_test)
matrix5 = np.histogram2d(y_predicted, y_test , bins=2)[0]
print(pd.DataFrame(matrix5, ['Down', 'Up'], ['Down', 'Up']))
print((100*np.mean(y_predicted != y_test)))

# Part h
# By comparing the test error rates from the four methods, we can see that logistic 
# regression as well as LDA have the smallest test error rates (37.5%)

# Part i #

# Question 11
auto = pd.read_csv('Auto.csv', na_values=['?'])
auto.dropna(inplace=True)
auto.reset_index(drop=True, inplace=True)

mpg = auto['mpg']
cylinder = auto['cylinders']
displacement = auto['displacement']
horsepower = auto['horsepower']
weight = auto['weight']
acceleration = auto['acceleration']
year = auto['year']
origin = auto['origin']

# Part a
mpg01 = (auto['mpg'] > auto['mpg'].median()).astype(np.float64)
auto = pd.concat([auto, mpg01.rename('mpg01')], axis=1)
display(auto.head())

# Part b
sns.pairplot(auto)
corr = auto.corr()

# cylinder vs mpg01
sns.boxplot(x = mpg01, y = cylinder)
plt.xlabel("mpg01")
plt.ylabel("cylinder")
plt.title("cylinder vs mpg01")

# displacement vs mpg01
sns.boxplot(x = mpg01, y = displacement)
plt.xlabel("mpg01")
plt.ylabel("displacement")
plt.title("displacement vs mpg01")

# horsepower vs mpg01
sns.boxplot(x = mpg01, y = horsepower)
plt.xlabel("mpg01")
plt.ylabel("horsepower")
plt.title("horsepower vs mpg01")

# weight vs mpg01
sns.boxplot(x = mpg01, y = weight)
plt.xlabel("mpg01")
plt.ylabel("weight")
plt.title("weight vs mpg01")

# Part c

# Training Set Index
samples = 392
row = np.random.choice([True, False], samples)
# Training Set
training = auto.loc[row]
# Test Set
test = auto.loc[~row]

# Part d

predictors = ['weight', 'cylinders', 'displacement', 'horsepower']
#predictors
x_training = training[predictors].values
y_training = training['mpg01'].values
x_test = test[predictors].values
y_test = test['mpg01'].values

# LDA
lda = LinearDiscriminantAnalysis(solver='lsqr', store_covariance=True)
lda.fit(x_training, y_training)
y_predicted = lda.predict(x_test)
print((100*np.mean(y_predicted != y_test))) #12.105263157894736

# Part e

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_training, y_training)
y_predicted = qda.predict(x_test)
print(100*np.mean(y_predicted != y_test)) #13.157894736842104

# Part f

x_training = sm.add_constant(training[predictors])
x_test = sm.add_constant(test[predictors])

logistic = sm.Logit(y_training, x_training)
result = logistic.fit()

predictions = result.predict(x_test)
y_predicted = np.array(predictions > 0.5, dtype=bool)
print(100*np.mean(y_predicted != y_test)) #12.631578947368421

# Part g

predictors = ['weight', 'cylinders', 'displacement', 'horsepower']
x_training =  training[predictors].values
y_training = training['mpg01'].values

x_test = test[predictors].values
y_test = test['mpg01'].values

# 20 KNN classifiers (1 through 20)
training_error_rate = np.zeros(20)
test_error_rate = np.zeros(20)
knn_values = np.arange(1,21)

for idx, k in enumerate(knn_values):
    # Construct a KNN classifier and fit
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_training, y_training)
    # obtain training error rate
    y_training_predicted = knn.predict(x_training)
    # obtain training error rate for k-value
    training_error_rate[idx] = np.mean(y_training_predicted != y_training)
    # Use the model on the held out test data
    y_test_predicted = knn.predict(x_test)
    #obtain error rate for k-value
    test_error_rate[idx] = np.mean(y_test_predicted != y_test)
print(test_error_rate)
min(test_error_rate) #0.15263157894736842
    
# Training and Test Error Rate Plot
fig, ax = plt.subplots(figsize=(10,5))
inverse_k = knn_values
ax.plot(inverse_k, test_error_rate, label='Test Errors');
ax.plot(inverse_k, training_error_rate, label='Training Errors')
ax.set_title('Training and Test Error Rate Plot')
ax.set_xlabel('Value of k')
ax.set_ylabel('Error Rate')
ax.set_xlim(xmin = 0, xmax = 3)
plt.legend()

# Question 12
    
# Part a
def Power():
    print(2**3)
Power()  

# Part b
def Power2(x,a):
    print(x**a)
Power2(2,2)
Power2(10,2)
Power2(4,3)

# Part c
def Power2(x,a):
    print(x**a)
Power2(10,3) #1000
Power2(8,17) #2251799813685248
Power2(131,3) #2248091

# Part d
def Power3(x,a):
    return(x**a)

# Part e
    
# f(x) = x^2
x = np.arange(1,10)
y = Power3(x,2)

fig, (graph1, graph2, graph3, graph4) = plt.subplots(1, 4, figsize=(20, 5))

# x vs y
graph1.plot(x,y)
graph1.set_title('x vs y')
graph1.set_xlabel('x')
graph1.set_ylabel('y')

# log(x) vs y
graph2.semilogx(x,y)
graph2.set_title('log(x) vs y')
graph2.set_xlabel('log(x)')
graph2.set_ylabel('y')

# x vs log(y)
graph3.semilogy(x,y)
graph3.set_title('x vs log(y)')
graph3.set_xlabel('x')
graph3.set_ylabel('log(y)')

# log(x) vs log(y)
graph4.loglog(x,y)
graph4.set_title('log(x) vs log(y)')
graph4.set_xlabel('log(x)')
graph4.set_ylabel('log(y)')

# Part f

def PlotPower(x,a):
    y = x**a
    fig, graph = plt.subplots(figsize=(8,5))
    graph.plot(x,y)
    graph.set_title('Question 12f Graph')
    graph.set_xlabel('x')
    graph.set_ylabel('x^3')
PlotPower(np.arange(1,11),3)

  
# Question 13

from sklearn.datasets import load_boston

boston = load_boston()

# What the variables are
print(boston['DESCR'])

# combine the predictors and responses for a dataframe
predictors = boston.data
response = boston.target
boston_data = np.column_stack([predictors,response])

# Get the column names of the data frame and create new dataframe
col_names = np.append(boston.feature_names, 'MEDV')
bos = pd.DataFrame(boston_data, columns = col_names)

crim = bos['CRIM']
zn = bos['ZN']
indus = bos['INDUS']
chas = bos['CHAS']
nox = bos['NOX']
rm = bos['RM']
age = bos['AGE']
dis = bos['DIS']
rad = bos['RAD']
tax = bos['TAX']
ptratio = bos['PTRATIO']
black = bos['B']
lstat = bos['LSTAT']
medv = bos['MEDV']

# Create a new column that displays "True" when crim > median(crim) and "False" otherwise
bos['crime_rate_med'] = pd.Series(bos.CRIM > bos.CRIM.median(), index=bos.index)
crime_rate_med = bos['crime_rate_med']
bos.head(3)

# Box Plots of NOX, RM, AGE, PTRATIO, LSTAT, MEDV vs Median Crime Rate

# NOX vs Median Crime Rate
sns.boxplot(x = crime_rate_med, y = nox)
plt.xlabel("Median Crime Rate")
plt.ylabel("NOX")
plt.title("NOX vs Median Crime Rate")

# RM vs Median Crime Rate
sns.boxplot(x = crime_rate_med, y = rm)
plt.xlabel("Median Crime Rate")
plt.ylabel("RM")
plt.title("RM vs Median Crime Rate")

# Age vs Median Crime Rate
sns.boxplot(x = crime_rate_med, y = age)
plt.xlabel("Median Crime Rate")
plt.ylabel("Age")
plt.title("Age vs Median Crime Rate")

# PTRATIO vs Median Crime Rate
sns.boxplot(x = crime_rate_med, y = ptratio)
plt.xlabel("Median Crime Rate")
plt.ylabel("PTRATIO")
plt.title("PTRATIO vs Median Crime Rate")

# LSTAT vs Median Crime Rate
sns.boxplot(x = crime_rate_med, y = lstat)
plt.xlabel("Median Crime Rate")
plt.ylabel("LSTAT")
plt.title("LSTAT vs Median Crime Rate")

# MEDV vs Median Crime Rate
sns.boxplot(x = crime_rate_med, y = medv)
plt.xlabel("Median Crime Rate")
plt.ylabel("MEDV")
plt.title("MEDV vs Median Crime Rate")

# Now let us explore the models!

# Split Data into Training and Test Sets
rows = np.random.choice([True, False], 506)
training = bos.loc[rows]
test = bos.loc[~rows]

# LDA Method

# NOX, PTRATIO and MEDV are the predictors associated with Median Crime Rate
predictors = ['NOX', 'PTRATIO', 'MEDV']

# Make Training/Test Matrices
x_training = training[predictors].values
x_test = test[predictors].values

# Obtain the training/test responses
y_training = training.crime_rate_med.values
y_test = test.crime_rate_med.values

# Model and Fit
lda = LDA(solver='lsqr',store_covariance=True)
lda.fit(x_training, y_training)

# Predict Test Data and Evaluate
y_predicted = lda.predict(x_test)
print((100*np.mean(y_predicted != y_test))) #17.479674796747968

# Logistic Regression Method

# Get Training/Test Predictors and Responses 
predictors = ['NOX', 'AGE', 'PTRATIO', 'LSTAT', 'MEDV']
x_training = sm.add_constant(training[predictors])
x_test = sm.add_constant(test[predictors])

y_training = training.crime_rate_med.values
y_test = test.crime_rate_med.values

# Model and Fit
logistic = sm.Logit(y_training, x_training)
result = logistic.fit()
print(result.summary())

# Predict Test Data and Evaluate
y_predicted = result.predict(x_test) > 0.5
print((100*np.mean(y_predicted != y_test))) #14.227642276422763


# KNN Method

# 20 KNN classifiers (1 through 20)
training_error_rate = np.zeros(20)
test_error_rate = np.zeros(20)
knn_values = np.arange(1,21)

for idx, k in enumerate(knn_values):
    # Construct a KNN classifier and fit
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_training, y_training)
    # obtain training error rate
    y_training_predicted = knn.predict(x_training)
    # obtain training error rate for k-value
    training_error_rate[idx] = np.mean(y_training_predicted != y_training)
    # Use the model on the held out test data
    y_test_predicted = knn.predict(x_test)
    #obtain error rate for k-value
    test_error_rate[idx] = np.mean(y_test_predicted != y_test)
print(test_error_rate)
min(test_error_rate) #0.18699186991869918
    
# Training and Test Error Rate Plot
fig, ax = plt.subplots(figsize=(10,5))
inverse_k = knn_values
ax.plot(inverse_k, test_error_rate, label='Test Errors');
ax.plot(inverse_k, training_error_rate, label='Training Errors')
ax.set_title('Training and Test Error Rate Plot')
ax.set_xlabel('Value of k')
ax.set_ylabel('Error Rate')
ax.set_xlim(xmin = 0, xmax = 20)
plt.legend()


