from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import feature_calculators
from sklearn.decomposition import PCA
from tsfresh import extract_features
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack
import sys

def mdate_to_pdate(mdate):
    pdate = datetime.fromordinal(int(mdate)) + timedelta(days=mdate%1) - timedelta(days = 366)
    pdate = pdate.strftime("%Y-%m-%d %H:%M:%S")
    return pdate

CGMDatenum = pd.read_csv("CGMDatenumLunchPat5.csv")
CGMSeries = pd.read_csv("CGMSeriesLunchPat5.csv")
CGMSeries = np.array(CGMSeries)

CGMSeries[np.isnan(CGMSeries)] = 0

#print(CGMSeries[1])

##CGMSeries = CGMSeries[np.logical_not(np.isnan(CGMSeries))]
x, y = CGMDatenum.shape
##res = ['']*x
# fea1 = ['']*x
# fea2 = ['']*x
# fea3 = ['']*x
#
# for i in range(len(CGMDatenum)):
#     fea1[i] = np.sum(np.abs(np.diff(CGMSeries[i])))
#     fea2[i] = feature_calculators.abs_energy(CGMSeries[i])
#     fea3[i] = feature_calculators.binned_entropy(CGMSeries[i],3)
#
# #for i in range(len(CGMDatenum)):
#     #print(i, fea2[i])
# fea_mat = np.zeros((len(fea1),3))
# x_coordinate = [ i for i in range(len(fea1))]
# for i in range(len(fea1)):
#     fea_mat[i] = [fea1[i], fea2[i], fea3[i]]
#     ##print(fea_mat[i])
# x = StandardScaler().fit_transform(fea_mat)
# X_normalized = preprocessing.normalize(fea_mat, norm='l2')
#
# # for i in range(len(X_normalized)):
# #     print(X_normalized[i])
#
# pca = PCA(n_components=5)
# dataset = pca.fit_transform(x)
# # pca.fit(x)
# print(dataset)
#
# plt.scatter(x_coordinate, fea3)
# plt.show()

v = ['']*(x*y)
w = ['']*(x*y)
datetime_object = ['']*(x*y)

for col in CGMDatenum.columns:
    CGMDatenum[col] = CGMDatenum[col].apply(lambda x: mdate_to_pdate(x) if pd.notnull(x) else x)
# k=0
# for i in range(x):
#     for j in range(y):
#         if j<9:
#             v[k] = CGMDatenum.iloc[i]['cgmDatenum_ '+str(j+1)]
#             w[k] = CGMSeries.iloc[i]['cgmSeries_ '+str(j+1)]
#         else:
#             v[k] = CGMDatenum.iloc[i]['cgmDatenum_'+str(j+1)]
#             w[k] = CGMSeries.iloc[i]['cgmSeries_'+str(j+1)]
#         k += 1
print(CGMDatenum)
# cleanedV = []
# cleanedW = []
#
# for i in range(len(w)):
#     if str(w[i]) != 'nan':
#         cleanedV.append(v[i])
#         cleanedW.append(w[i])
#
# abs = []*(len(cleanedV)-1)
#
# for i in range(len(cleanedV)-1):
#     abs[i] = float(cleanedV[i+1])-float(cleanedV[i])
#
# print(abs)
#
# ##for i in range(len(cleanedV)):
# ##    print(cleanedV[i],cleanedW[i])
# ##
# datetime_object = ['']*len(cleanedV)
# for i in range(len(cleanedV)):
#     datetime_object[i] = str(datetime.strptime(str(cleanedV[i]),'%Y-%m-%d %H:%M:%S').date())
# ##for i in range(len(cleanedV)):
# ##    print(cleanedV[i])
#
# uni_date = np.unique(datetime_object, return_counts=True)
# for i in range(len(uni_date[0])):
#     if uni_date[1][i] == 0:
#         np.delete(uni_date, i)
#
#
# sum_cgm = [0]*(len(uni_date[0]))
# mean = [0]*(len(uni_date[0]))
# matT = np.empty((x,y), dtype = float)
#
# for i in range(len(uni_date[0])):
#     k = 0
#     for j in range(len(cleanedV)):
#         if uni_date[0][i] in cleanedV[j]:
#             sum_cgm[i] += cleanedW[j]
#             matT[i][k] = cleanedW[j]
#             k += 1
# ##print("Date         Mean")
# ##for i in range(len(matT)):
# ##    for j in range(20,len(matT[0])):
# ##        if matT[i][j] > 1 and matT[i][j] < 400:
# ##            sys.stdout.write(str(matT[i][j])+'\t')
# ##    print('')
# for i in range(len(sum_cgm)):
#     mean[i] = float(sum_cgm[i]/uni_date[1][i])
#     ##print(uni_date[0][i], mean[i])
#
# model = SimpleExpSmoothing(cleanedW)
# model_fit = model.fit()
# print(model_fit.summary())
#
# mean_fft = sp.fftpack.fft(mean)
# mean_psd = np.abs(mean_fft) ** 2
# fftfreq = sp.fftpack.fftfreq(len(mean_psd))
#
# plt.plot(cleanedV,cleanedW)
# ##plt.xlim(CGMDatenum.iloc[0]['cgmDatenum_ 1'],CGMDatenum.iloc[0]['cgmDatenum_31'])
# plt.xticks(rotation=90)
# ##plt.show()
