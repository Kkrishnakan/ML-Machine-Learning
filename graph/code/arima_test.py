#-*- coding: utf-8 -*-


import pandas as pd

disfile = '../data/arima_data.xls'
forecastnum = 5

data = pd.read_excel(disfile,index_col=u'date')

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
data.plot()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data).show()


from statsmodels.tsa.stattools import adfuller as ADF
print(u'The ADF test result of the original sequence is：',ADF(data[u'Sales volume']))


D_data = data.diff().dropna()
D_data.columns = [u'Sales differential']
D_data.plot()
plt.show()
plot_acf(D_data).show()
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data).show()
print(u'The ADF test result of the difference sequence is：',ADF(D_data[u'Sales differential']))


from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The white noise test result of the difference sequence is：',acorr_ljungbox(D_data,lags=1))

from statsmodels.tsa.arima_model import ARIMA

pmax = int(len(D_data)/10)
qmax = int(len(D_data)/10)
bic_matrix = [] 
for p in range(pmax+1):
    tmp = []
    for q in range(qmax):
        try:
            tmp.append(ARIMA(data,(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix = pd.DataFrame(bic_matrix)
p,q = bic_matrix.stack().idxmin()
print(u'The minimum p and q values ​​of BIC：%s,%s'%(p,q))
model = ARIMA(data,(p,1,q)).fit()
model.summary2()
model.forecast(5)
