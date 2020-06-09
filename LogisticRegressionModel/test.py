import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tsfresh.feature_extraction import feature_calculators
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
from random import seed
from random import randrange
import pywt
import sys

def readCSV(csv_filepath = 'dataset1.csv'):
    testData = pd.read_csv(csv_filepath, header=None)
    return testData

def pred_values(beta, X):
    pred_prob = logistic_func(beta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    return np.squeeze(pred_value)

def logistic_func(beta, X):
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))

beta = pd.read_csv('beta.csv', header=None)
beta = np.asarray(beta)
csv_filepath = sys.argv[1]
testData = readCSV(csv_filepath)
# testData = testData.dropna()
nrows = testData.shape[0]
ncolumns = testData.shape[1]

"""## Discrete Wavelet Transform
The coefficient of DWT sort of approximates the signal
"""

dwt_coeff = []

for i in range(nrows):
    (data, coeff_d) = pywt.dwt(testData.iloc[i,:], 'db2')
    (data, coeff_d) = pywt.dwt(data, 'db2') # Apply 2 levels of approximation
    dwt_coeff.append(coeff_d)

index = [i for i in range(nrows)]

"""## Kurtosis
Measures the skewness of the series data
Essentially how much do the values deviate from the normal values
"""

kurtosis = []

for i in range(nrows):
    kurtosis.append(feature_calculators.kurtosis(testData.iloc[i,:]))

"""## Large Amplitude of plasma Glucose Excursions (LAGE=xmax−xmin)"""

LAGE = np.zeros(nrows)

for i in range(int(nrows/2)):
    LAGE[i] = np.max(testData.iloc[i,:]) - np.min(testData.iloc[i,:])

"""## Low and High Blood Glucose Index"""

CGM_new = 1.509*(np.log(testData)**1.084 - 5.381)
risk = 10*CGM_new**2

LBGI = np.zeros(nrows) ##Feature 2: Low Blood Glucose Index(LBGI)
HBGI = np.zeros(nrows) ##Feature 3: High Blood Glucose Index(HBGI)

for i in range(nrows):
    k=0
    for j in range(ncolumns):
        if CGM_new.iloc[i][j] < 0:
            LBGI[i] += risk.iloc[i][j]
            k+=1
        elif CGM_new.iloc[i][j] > 0:
            HBGI[i] += risk.iloc[i][j]
            k+=1
    if(k != 0):
        LBGI[i] = LBGI[i]/k
        HBGI[i] = HBGI[i]/k

"""## Continuous overall net glycemic action(CONGA)"""

CONGA = np.zeros(nrows)
for i in range(nrows):
    temp = []
    for j in range(ncolumns-6):
        if  not pd.isnull(testData.iloc[i][j+6]) and not pd.isnull(testData.iloc[i][j]):
            temp.append(testData.iloc[i][j+6] - testData.iloc[i][j])
    if temp == []:
        CONGA[i] = 0
    else:
        CONGA[i] = np.std(temp)

"""##Create Normalized Feature Matrix"""

fea_mat = ['']*(nrows)

for i in range(nrows):
    fea_mat[i] = np.concatenate((dwt_coeff[i], np.asarray([kurtosis[i], LAGE[i], LBGI[i], HBGI[i], CONGA[i]])))

fea_mat = np.asarray(fea_mat)

nor_mat = preprocessing.normalize(fea_mat,axis=0, norm='max')

pca1 = PCA().fit(nor_mat.data)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
pca = PCA(n_components=5)#Top 5 Features selected
X_test = pca.fit_transform(nor_mat)

np.random.shuffle(X_test)

X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test))

# predicted labels
y_pred = pred_values(beta, X_test)

for i in range(len(y_pred)):
    print('Test sample no.: {0}, Predicted Label: {1}'.format((i+1), y_pred[i]))
