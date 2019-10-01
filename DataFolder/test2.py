from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack

def mdate_to_pdate(mdate):
    pdate = datetime.fromordinal(int(mdate)) + timedelta(days=mdate%1) - timedelta(days = 366)
    pdate = pdate.strftime("%Y-%m-%d %H:%M:%S")
    return pdate

CGMDatenum = pd.read_csv("CGMDatenumLunchPat4.csv")
CGMSeries = pd.read_csv("CGMSeriesLunchPat4.csv")

x, y = CGMDatenum.shape

v = ['']*(x*y)
w = ['']*(x*y)
datetime_object = ['']*(x*y)

for col in CGMDatenum.columns:
    CGMDatenum[col] = CGMDatenum[col].apply(lambda x: mdate_to_pdate(x) if pd.notnull(x) else x)
k=0
for i in range(x):
    for j in range(y):
        if j<9:
            v[k] = CGMDatenum.iloc[i]['cgmDatenum_ '+str(j+1)]
            w[k] = CGMSeries.iloc[i]['cgmSeries_ '+str(j+1)]
        else:
            v[k] = CGMDatenum.iloc[i]['cgmDatenum_'+str(j+1)]
            w[k] = CGMSeries.iloc[i]['cgmSeries_'+str(j+1)]
        k += 1

cleanedV = []
cleanedW = []

for i in range(len(w)):
    if str(w[i]) != 'nan':
        cleanedV.append(v[i])
        cleanedW.append(w[i])

##for i in range(len(cleanedV)):
##    print(cleanedV[i],cleanedW[i])
##        
datetime_object = ['']*len(cleanedV)
for i in range(len(cleanedV)):
    datetime_object[i] = str(datetime.strptime(str(cleanedV[i]),'%Y-%m-%d %H:%M:%S').date())

uni_date = np.unique(datetime_object, return_counts=True)
for i in range(len(uni_date[0])):
    if uni_date[1][i] == 0:
        np.delete(uni_date, i)


sum_cgm = [0]*(len(uni_date[0]))
mean = [0]*(len(uni_date[0]))
            
for i in range(len(uni_date[0])):
    for j in range(len(cleanedV)):
        if uni_date[0][i] in cleanedV[j]:
            sum_cgm[i] += cleanedW[j]

print("Date         Mean")   

for i in range(len(sum_cgm)):
    mean[i] = float(sum_cgm[i]/uni_date[1][i])
    print(uni_date[0][i], mean[i])

mean_fft = sp.fftpack.fft(mean)
mean_psd = np.abs(mean_fft) ** 2
fftfreq = sp.fftpack.fftfreq(len(mean_psd))

plt.plot(uni_date[0],mean)
##plt.xlim(CGMDatenum.iloc[0]['cgmDatenum_ 1'],CGMDatenum.iloc[0]['cgmDatenum_31'])
plt.xticks(rotation=90)
plt.show()
