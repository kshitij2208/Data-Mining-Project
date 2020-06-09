from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pywt
from tsfresh.feature_extraction import feature_calculators
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

def read_dataset(csv_filepath):

    cgm = pd.read_csv(csv_filepath, usecols=list(range(30)) , names=list(range(30)) )

    print(cgm.head())
    print(cgm.shape)
    cgm = cgm.astype('float')#.values

    return cgm

def calculateNormFeatures(cgm):

    nrows = cgm.shape[0]
    ncolumns = cgm.shape[1]

    degree = 7
    polyCoeff = np.zeros((nrows,degree+1))
    for i in range(nrows): polyCoeff[i,:] = np.polyfit(range(0,150,5),cgm.iloc[i,:], deg=degree)
    #print(polyCoeff)

    fft = np.zeros((nrows,ncolumns))
    for i in range(nrows):
        temp = np.fft.fft(cgm.iloc[i,:])
        fft[i,:] = np.absolute(temp)
    #print(fft)

    dwt_coeff = []
    for i in range(nrows):

        (data, coeff_d) = pywt.dwt(cgm.iloc[i,:], 'db2')
        (data, coeff_d) = pywt.dwt(data, 'db2') # Apply 2 levels of approximation
        #print(len(coeff_d))
        dwt_coeff.append(coeff_d)

    kurtosis = []
    for i in range(nrows): kurtosis.append(feature_calculators.kurtosis(cgm.iloc[i,:]))
    #print(kurtosis)

    LAGE = np.zeros(nrows)
    for i in range(nrows):
        LAGE[i] = np.max(cgm.iloc[i,:]) - np.min(cgm.iloc[i,:])
    #print(lage)

    TIR = np.zeros(nrows)
    for i in range(nrows):
        for j in range(1,ncolumns):
            if cgm.iloc[i][j] > 80.0 and cgm.iloc[i][j] < 180.0:
                TIR[i] += cgm.iloc[i][j] - cgm.iloc[i][j-1]
        TIR[i] = TIR[i]*-1

    CONGA = np.zeros(nrows)
    for i in range(nrows):
        temp = []
        for j in range(ncolumns-6):
            if  not pd.isnull(cgm.iloc[i][j+6]) and not pd.isnull(cgm.iloc[i][j]):
                temp.append(cgm.iloc[i][j+6] - cgm.iloc[i][j])
        if temp == []:
            CONGA[i] = 0
        else:
            CONGA[i] = np.std(temp)

    fea_mat = ['']*nrows
    for i in range(nrows):
        fea_mat[i] = np.concatenate((polyCoeff[i],fft[i,:8],dwt_coeff[i], np.asarray([kurtosis[i], LAGE[i], TIR[i], CONGA[i]])))

    fea_mat = np.asarray(fea_mat)
    nor_mat = preprocessing.normalize(fea_mat,axis=0, norm='max')

    return nor_mat

def main():

    cgm = read_dataset(sys.argv[1])

    fea_mat = calculateNormFeatures(cgm)

    pca = pickle.load(open('yash_pca', 'rb'))
    dataset = pca.transform(fea_mat)

    clf = pickle.load(open('yash_classifier', 'rb'))
    y_pred = clf.predict(dataset)

    print(y_pred)

if __name__ == '__main__' :
    main()