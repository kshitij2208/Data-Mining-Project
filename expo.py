
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import numpy as np
##%matplotlib inline
plt.style.use('Solarize_Light2')

r = requests.get('https://datamarket.com/api/v1/list.json?ds=22qx')
jobj = json.loads(r.text[18:-1])
data = jobj[0]['data']
print(data)
