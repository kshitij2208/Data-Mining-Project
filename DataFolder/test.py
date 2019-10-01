import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
from numpy import genfromtxt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
CGMDatenum_filepath = "CGMDatenumLunchPat1.csv"
CGMSeries_filepath = "CGMSeriesLunchPat1.csv"

CGMDatenum = pd.read_csv(CGMDatenum_filepath)
CGMSeries = pd.read_csv(CGMSeries_filepath)

first = CGMDatenum.loc[0]

print(first)

##mean = []
##timestamp = []
##for i in range(31):
##    if i<9:
##        mean.append(np.mean(CGMSeries['cgmSeries_ '+str(i+1)]))
##    else:
##        mean.append(np.mean(CGMSeries['cgmSeries_'+str(i+1)]))
##    timestamp.append(i)
##
### Line chart 
##plt.figure(figsize=(12,6))
###sns.lineplot(data=CGMSeries['cgmSeries_ 1'])
##
##### plotting CGMSeries against time
####sns.tsplot(data=CGMSeries['cgmSeries_ 1'], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####
####sns.tsplot(data=CGMSeries['cgmSeries_ 1'], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level", interpolate=False)
##
### plotting multiple CGMSeries against time
##sns.tsplot(data=mean, time=timestamp, value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 10']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 11']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 12']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 13']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 14']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 15']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 16']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 17']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 18']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 19']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 20']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 21']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 22']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 23']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 24']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 25']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 26']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 27']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 28']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 29']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
####sns.tsplot(data=[CGMSeries['cgmSeries_ 30']], time=CGMDatenum['cgmDatenum_ 1'], value="CGM level")
##
##
##plt.show()
