from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pywt
from tsfresh.feature_extraction import feature_calculators
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
import math
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    return TP, FP, TN, FN

def logistic_func(beta, X):
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))

def log_gradient(beta, X, y):
    first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X)
    return final_calc

def cost_func(beta, X, y):
    log_func_v = logistic_func(beta, X)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)

def grad_desc(X, y, beta, lr=.01, converge_change=.001):
    cost = cost_func(beta, X, y)
    change_cost = 1
    num_iter = 1

    while(change_cost > converge_change):
        old_cost = cost
        beta = beta - (lr * log_gradient(beta, X, y))
        cost = cost_func(beta, X, y)
        change_cost = old_cost - cost
        num_iter += 1

    return beta, num_iter

def pred_values(beta, X):
    pred_prob = logistic_func(beta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    return np.squeeze(pred_value)

def cross_validation_split(dataset, folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

nomeal1 = pd.read_csv('data/Nomeal1.csv', header=None)
nomeal1 = nomeal1.dropna()
nrows1 = nomeal1.shape[0]
ncolumns1 = nomeal1.shape[1]
nomeal2 = pd.read_csv('data/Nomeal2.csv', header=None)
nomeal2 = nomeal2.dropna()
nrows2 = nomeal2.shape[0]
ncolumns2 = nomeal2.shape[1]
nomeal3 = pd.read_csv('data/Nomeal3.csv', header=None)
nomeal3 = nomeal3.dropna()
nrows3 = nomeal3.shape[0]
ncolumns3 = nomeal3.shape[1]
nomeal4 = pd.read_csv('data/Nomeal4.csv', header=None)
nomeal4 = nomeal4.dropna()
nrows4 = nomeal4.shape[0]
ncolumns4 = nomeal4.shape[1]
nomeal5 = pd.read_csv('data/Nomeal5.csv', header=None)
nomeal5 = nomeal5.dropna()
nrows5 = nomeal5.shape[0]
ncolumns5 = nomeal5.shape[1]

meal1 = pd.read_csv('data/mealData1.csv', header=None)
meal1 = meal1.dropna()
nrows6 = meal1.shape[0]
ncolumns6 = meal1.shape[1]
meal2 = pd.read_csv('data/mealData2.csv', header=None)
meal2 = meal2.dropna()
nrows7 = meal2.shape[0]
ncolumns7 = meal2.shape[1]
meal3 = pd.read_csv('data/mealData3.csv', header=None)
meal3 = meal3.dropna()
nrows8 = meal3.shape[0]
ncolumns8 = meal3.shape[1]
meal4 = pd.read_csv('data/mealData4.csv', header=None)
meal4 = meal4.dropna()
nrows9 = meal4.shape[0]
ncolumns9 = meal4.shape[1]
meal5 = pd.read_csv('data/mealData5.csv', header=None)
meal5 = meal5.dropna()
nrows10 = meal5.shape[0]
ncolumns10 = meal5.shape[1]

nomeal_len = nrows1+nrows2+nrows3+nrows4+nrows5
meal_len= nrows6+nrows7+nrows8+nrows9+nrows10
combined_mat = np.concatenate((nomeal1,nomeal2,nomeal3,nomeal4,nomeal5,meal1,meal2,meal3,meal4,meal5), axis=0)

"""## Discrete Wavelet Transform
The coefficient of DWT sort of approximates the signal
"""
combined_mat = pd.DataFrame(combined_mat)
combined_mat.to_csv("dataset1.csv",header=None,index =None)

dwt_coeff = []

for i in range(meal_len+nomeal_len):
    (data, coeff_d) = pywt.dwt(combined_mat.iloc[i,:], 'db2')
    (data, coeff_d) = pywt.dwt(data, 'db2') # Apply 2 levels of approximation
    dwt_coeff.append(coeff_d)

index = [i for i in range(meal_len+nomeal_len)]

"""## Kurtosis
Measures the skewness of the series data
Essentially how much do the values deviate from the normal values
"""

kurtosis = []

for i in range(nomeal_len+meal_len):
    kurtosis.append(feature_calculators.kurtosis(combined_mat.iloc[i,:]))

"""## Large Amplitude of plasma Glucose Excursions (LAGE=xmax−xmin)"""

LAGE = np.zeros(meal_len+nomeal_len)

for i in range(nomeal_len):
    LAGE[i] = np.max(combined_mat.iloc[i,:]) - np.min(combined_mat.iloc[i,:])

"""## Low and High Blood Glucose Index"""

CGM_new = 1.509*(np.log(combined_mat)**1.084 - 5.381)
risk = 10*CGM_new**2

LBGI = np.zeros(meal_len+nomeal_len) ##Feature 2: Low Blood Glucose Index(LBGI)
HBGI = np.zeros(meal_len+nomeal_len) ##Feature 3: High Blood Glucose Index(HBGI)

for i in range(nomeal_len+meal_len):
    k=0
    for j in range(ncolumns1):
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

CONGA = np.zeros(meal_len+nomeal_len)
for i in range(nomeal_len+meal_len):
    temp = []
    for j in range(ncolumns1-6):
        if  not pd.isnull(combined_mat.iloc[i][j+6]) and not pd.isnull(combined_mat.iloc[i][j]):
            temp.append(combined_mat.iloc[i][j+6] - combined_mat.iloc[i][j])
    if temp == []:
        CONGA[i] = 0
    else:
        CONGA[i] = np.std(temp)

"""##Create Normalized Feature Matrix"""

fea_mat = ['']*(nomeal_len+meal_len)

for i in range(nomeal_len+meal_len):
    fea_mat[i] = np.concatenate((dwt_coeff[i], np.asarray([kurtosis[i], LAGE[i], LBGI[i], HBGI[i], CONGA[i]])))

fea_mat = np.asarray(fea_mat)

nor_mat = preprocessing.normalize(fea_mat,axis=0, norm='max')

pca1 = PCA().fit(nor_mat.data)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
pca = PCA(n_components=5)#Top 5 Features selected
dataset = pca.fit_transform(nor_mat)

nor_mat_new = np.zeros((nomeal_len+meal_len,6))
for i in range(nomeal_len+meal_len):
    if(i < nomeal_len):
        nor_mat_new[i] = np.append(dataset[i], 0)
    else:
        nor_mat_new[i] = np.append(dataset[i], 1)

#Number of Folds
k = 10

#k-fold cross validation
folds = cross_validation_split(nor_mat_new, k)
folds = np.asarray(folds)

accuracy_list = []
recall_list = []
precision_list = []
f1_score_list = []
beta_list = []
for i in range(k):
    X_train = []
    y_train= []
    y_test = []
    X_test = []

    for j in range(k):
        if(j!=i):
            for l in range(len(folds[j])):
                X_train.append(folds[j,l,range(0,5)])
                y_train.append(folds[j,l,5])
    for l in range(len(folds[i])):
        X_test.append(folds[i,l,range(0,5)])
        y_test.append(folds[i,l,5])
    X_train = np.asarray(X_train)
    y_train= np.asarray(y_train)
    y_test = np.asarray(y_test)
    X_test = np.asarray(X_test)
    X_train = np.hstack((np.matrix(np.ones(X_train.shape[0])).T, X_train))
    X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test))

    beta = np.matrix(np.zeros(X_train.shape[1]))
    beta, num_iter = grad_desc(X_train, y_train, beta)

    count0 = 0
    count1 = 0
    # predicted labels
    y_pred = pred_values(beta, X_test)

    for l in range(len(y_pred)):
        if y_test[l] == 0.0 and y_test[l] == y_pred[l]:
            count0 += 1
        elif y_test[l] == 1.0 and y_test[l] == y_pred[l]:
            count1 += 1
    TP, FP, TN, FN = perf_measure(y_test, y_pred)

    accuracy = (TP+TN)/len(y_test)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1_score = 2*recall*precision/(recall+precision)
    # print("Accuracy Iteration No. "+str(i)+": "+str(accuracy))
    # print("Recall Iteration No. "+str(i)+": "+str(recall))
    # print("Precision Iteration No. "+str(i)+": "+str(precision))
    # print("F1 Score Iteration No. "+str(i)+": "+str(f1_score))
    accuracy_list.append(accuracy)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_score_list.append(f1_score)

    # estimated beta values and number of iterations
    # print("Estimated regression coefficients:", beta)
    # print("No. of iterations:", num_iter)
    beta_list.append(beta)

# enumerate splits
print("\nNumber of folds: "+str(k))
print("\nOverall Accuracy: "+str(np.mean(accuracy_list)))
print("Overall Recall: "+str(np.mean(recall_list)))
print("Overall Precision: "+str(np.mean(precision_list)))
print("Overall F1 Score: "+str(np.mean(f1_score_list)))

acc_max = np.max(accuracy)
index = np.where(accuracy == acc_max)
beta_max = beta_list[index[0][0]]
pd.DataFrame(beta_max).to_csv("beta.csv",header=None,index =None)
