#-*- coding: utf-8 -*-

import pandas as pd
datafile = '../data/discretization_data.xls'
data = pd.read_excel(datafile)
data = data[u'coefficient'].copy()

k = 4

d1 = pd.cut(data,k,labels=range(k))


w = [1.0*i/k for i in range(k+1)]
w = data.describe(percentiles = w)[4:4+k+1]
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(data,w,labels=range(k))

from sklearn.cluster import KMeans
kmodel = KMeans(n_clusters=k)
kmodel.fit(data.values.reshape((len(data),1)))
c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
#w = pd.rolling_mean(c,2).iloc[1:]
w = pd.DataFrame.rolling(c,center = False,window = 2).mean().iloc[1:]
w = [0] + list(w[0]) + [data.max()]
d3 = pd.cut(data,w,labels=range(k))

def cluster_plot(d,k):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8,3))
    for j in range(0,k):
        plt.plot(data[d==j],[i for i in d[d==j]],'o')

    plt.ylim(-0.5,k-0.5)
    return plt

cluster_plot(d1,k).show()
cluster_plot(d2,k).show()
cluster_plot(d3,k).show()