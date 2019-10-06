import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

CGMDatenum_1 = pd.read_csv("CGMDatenumLunchPat1.csv")
CGMSeries_1 = pd.read_csv("CGMSeriesLunchPat1.csv")
CGMDatenum_2 = pd.read_csv("CGMDatenumLunchPat2.csv")
CGMSeries_2 = pd.read_csv("CGMSeriesLunchPat2.csv")
CGMDatenum_3 = pd.read_csv("CGMDatenumLunchPat3.csv")
CGMSeries_3 = pd.read_csv("CGMSeriesLunchPat3.csv")
CGMDatenum_4 = pd.read_csv("CGMDatenumLunchPat4.csv")
CGMSeries_4 = pd.read_csv("CGMSeriesLunchPat4.csv")
CGMDatenum_5 = pd.read_csv("CGMDatenumLunchPat5.csv")
CGMSeries_5 = pd.read_csv("CGMSeriesLunchPat5.csv")

x1, y1 = CGMSeries_5.shape

##Feature 1: Large Amplitude of plasma Glucose Excursions (LAGE=xmax−xmin)
fea_1 = ['']*x1
for i in range(x1):
    fea_1[i] = np.max(CGMSeries_5.loc[i]) - np.min(CGMSeries_5.loc[i])

CGM_new = 1.509*(np.log(CGMSeries_5)**1.084 - 5.381)
risk = 10*CGM_new**2

fea_2 = [0]*x1 ##Feature 2: Low Blood Glucose Index(LBGI)
fea_3 = [0]*x1 ##Feature 3: High Blood Glucose Index(HBGI)

for i in range(x1):
    k=0
    for j in range(y1):
        if CGM_new.iloc[i][j] < 0:
            fea_2[i] += risk.iloc[i][j]
            k+=1
        elif CGM_new.iloc[i][j] > 0:
            fea_3[i] += risk.iloc[i][j]
            k+=1
    if(k != 0):
        fea_2[i] = fea_2[i]/k
        fea_3[i] = fea_3[i]/k

##Feature 4: Time in Range(TIR) (Range 80 mg/dL - 170 mg/dL)
fea_4 = [0]*x1
for i in range(x1):
    for j in range(1,y1):
        if CGMSeries_5.iloc[i][j] > 70.0 and CGMSeries_5.iloc[i][j] < 180.0:
            fea_4[i] += CGMDatenum_5.iloc[i][j] - CGMDatenum_5.iloc[i][j-1]
    fea_4[i] = fea_4[i]*-1

##Feature 5: Continuous overall net glycemic action(CONGA)
fea_5 = [0]*x1
for i in range(x1):
    temp = []
    for j in range(y1-6):
        if  not pd.isnull(CGMSeries_5.iloc[i][j+6]) and not pd.isnull(CGMSeries_5.iloc[i][j]):
            temp.append(CGMSeries_5.iloc[i][j+6] - CGMSeries_5.iloc[i][j])
    if temp == []:
        fea_5[i] = 0
    else:
        fea_5[i] = np.std(temp)

##Feature Matrix
fea_mat = ['']*x1
for i in range(len(fea_1)):
    fea_mat[i] = [fea_1[i], fea_2[i], fea_3[i], fea_4[i], fea_5[i]]

for i in range(len(fea_1)):
    print(i, fea_mat[i])

nor_mat = preprocessing.normalize(fea_mat,axis=0, norm='max')
print(nor_mat)

##Principal Component Analysis
pca = PCA(n_components=5)#Top 5 Features selected
dataset = pca.fit_transform(nor_mat)
for i in range(len(dataset)):
    print(dataset[i])


fig = plt.figure()
ax1 = fig.add_subplot(111)
x_coordinate = [ i for i in range(len(fea_1))]
ax1.scatter(x_coordinate, nor_mat[:,0], s=10, c='b', marker="s", label='LAGE')
ax1.scatter(x_coordinate, nor_mat[:,1], s=10, c='r', marker="o", label='LBGI')
ax1.scatter(x_coordinate, nor_mat[:,2], s=10, c='y', marker="s", label='HBGI')
ax1.scatter(x_coordinate, nor_mat[:,3], s=10, c='g', marker="o", label='TIR')
ax1.scatter(x_coordinate, nor_mat[:,4], s=10, c='k', marker="o", label='CONGA')
plt.ylabel("Feature Values")
plt.xlabel("Timestamp")
plt.legend(loc='upper left');
plt.show()
