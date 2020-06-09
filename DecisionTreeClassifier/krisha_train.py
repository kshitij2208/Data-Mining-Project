from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pywt
import pickle
import sys
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



def read_dataset():
    meal = pd.read_csv('data/mealData1.csv',error_bad_lines=False,usecols=list(range(30)), names=list(range(30)))
    for i in range(2,6):
        tmp = pd.read_csv('data/mealData'+str(i)+'.csv',error_bad_lines=False,usecols=list(range(30)), names=list(range(30)))
        meal = meal.append(tmp)

    label = ([1] * meal.shape[0])


    nomeal = pd.read_csv('data/Nomeal1.csv',error_bad_lines=False,usecols=list(range(30)), names=list(range(30)))
    for i in range(2,6):
        tmp1 = pd.read_csv('data/Nomeal'+str(i)+'.csv',error_bad_lines=False,usecols=list(range(30)), names=list(range(30)))
        nomeal = nomeal.append(tmp1)

    label1 = ([0] * nomeal.shape[0])
    meal = meal.append(nomeal)
    label.extend(label1)
    print(label)
    return meal,label


def calculateNormFeature(meal):
    nrows = meal.shape[0]
    ncolumns = meal.shape[1]
    degree = 8
    polyCoeff = np.zeros((nrows,degree+1))
    for i in range(nrows): polyCoeff[i,:] = np.polyfit(range(0,150,5),meal.iloc[i,:], deg=degree)
    #print(polyCoeff)

    fft = np.zeros((nrows,ncolumns))
    for i in range(nrows):
        temp = np.fft.fft(meal.iloc[i,:])
        fft[i,:] = np.absolute(temp)
    #print(fft)

    dwt_coeff = []
    for i in range(nrows):

        (data, coeff_d) = pywt.dwt(meal.iloc[i,:], 'db2')
        (data, coeff_d) = pywt.dwt(data, 'db2') # Apply 2 levels of approximation
        #print(len(coeff_d))
        dwt_coeff.append(coeff_d)
   

    LAGE = np.zeros(nrows)
    for i in range(nrows):
        LAGE[i] = np.max(meal.iloc[i,:]) - np.min(meal.iloc[i,:])
    #print(lage)
   
    TIR = np.zeros(nrows)
    for i in range(nrows):
        for j in range(1,ncolumns):
            if meal.iloc[i][j] > 80.0 and meal.iloc[i][j] < 180.0:
                TIR[i] += meal.iloc[i][j] - meal.iloc[i][j-1]
        TIR[i] = TIR[i]*-1

    CONGA = np.zeros(nrows)
    for i in range(nrows):
        temp = []
        for j in range(ncolumns-6):
            if  not pd.isnull(meal.iloc[i][j+6]) and not pd.isnull(meal.iloc[i][j]):
                temp.append(meal.iloc[i][j+6] - meal.iloc[i][j])
        if temp == []:
            CONGA[i] = 0
        else:
            CONGA[i] = np.std(temp)

    fea_mat = ['']*nrows
    for i in range(nrows):
        fea_mat[i] = np.concatenate((polyCoeff[i],fft[i,:8],dwt_coeff[i], np.asarray([LAGE[i],TIR[i], CONGA[i]])))

    fea_mat = np.asarray(fea_mat)
    nor_mat = preprocessing.normalize(fea_mat,axis=0, norm='max')
    return nor_mat


def main():
    meal, label = read_dataset()
    fea_mat = calculateNormFeature(meal)

    pca = PCA(n_components=10)#Top 10 Features selected
    dataset = pca.fit_transform(fea_mat)
    X_train,X_test,y_train,y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)

    """
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lda = LDA(n_components=1)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    """


    #classifier = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4,3,2),random_state=1)

    classifier = tree.DecisionTreeClassifier(random_state=2, max_depth=2, min_samples_split=4)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    pickle.dump(classifier, open('krisha_classifier_new', 'wb'))
    pickle.dump(pca, open('krisha_pca_new', 'wb'))

if __name__ == '__main__' :
    main()
