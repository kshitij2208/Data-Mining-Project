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

def read_dataset():

    cgm = pd.read_csv('data/mealData1.csv', usecols=list(range(30)) , names=list(range(30)) )
    labels = []
    for i in range(2,6):
        tmp = pd.read_csv('data/mealData'+str(i)+'.csv', usecols=list(range(30)) , names=list(range(30)))
        cgm = cgm.append(tmp)

    labels.extend([1]*cgm.shape[0])

    for i in range(1,6):
        tmp = pd.read_csv('data/Nomeal'+str(i)+'.csv', usecols=list(range(30)) , names=list(range(30)))
        cgm = cgm.append(tmp)
        labels.extend([0]*tmp.shape[0])

    print(cgm.head())
    print(cgm.shape)
    cgm = cgm.astype('float')#.values

    return cgm, labels

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

    cgm, labels = read_dataset()

    fea_mat = calculateNormFeatures(cgm)


    """
    pca1 = PCA().fit(nor_mat.data)
    plt.plot(np.cumsum(pca1.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    #plt.show()
    """

    pca = PCA(n_components=10)#Top 10 Features selected
    pca.fit(fea_mat)
    dataset = pca.transform(fea_mat)
    #print(pca.explained_variance_)
    #print(pca.explained_variance_ratio_)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

    clf = RandomForestClassifier(n_estimators=250, random_state=0,)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Confusion Matrix:')
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print('Accuracy:'+ str(accuracy_score(y_test, y_pred)))

    pickle.dump(clf, open('yash_classifier_new', 'wb'))
    pickle.dump(pca, open('yash_pca_new', 'wb'))

if __name__ == '__main__' :
    main()
