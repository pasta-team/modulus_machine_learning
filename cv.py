#!/home/jjs/miniconda3/bin/python

import os
import pickle,linecache,math
import numpy as np
from pasta.element import Element
import xgboost as xgb
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve,validation_curve,GridSearchCV
from random import shuffle
from math import sqrt
import matplotlib.pyplot as plt

def powermean(w,x,n):
    if n==0:
        res = 1.0
        su = sum(w)
        for i,j in zip(w,x):
            res *= pow(j,i/su)
        return res
    a = 0.0
    b = 0.0
    for i,j in zip(w,x):
        a += i
        b += i * pow(j,n)
    return pow(b/a,1/n)

def std(w,x):
    li = []
    for c,n in zip(w,x):
        li.extend([n]*c)
    return np.std(np.array(li))

def watch(Y1,y1_rbf):
    len1 = len(y1_rbf)
    acc1 = []
    for i in range(len1):
        acc1.append(math.fabs(math.exp(y1_rbf[i]-Y1[i])-1))
    num5 = 0
    num10 = 0
    num20 = 0
    num30 = 0
    for i in acc1:
        if i <= 0.05:
            num5 += 1
            num10 += 1
            num20 += 1
            num30 += 1
        elif 0.05 < i <= 0.1:
            num10 += 1
            num20 += 1
            num30 += 1
        elif 0.1 < i <= 0.2:
            num20 += 1
            num30 += 1
        elif 0.2<i<= 0.3:
            num30 += 1

    return (num5/len1,num10/len1,num20/len1,num30/len1)

def rmse(Y1,y1_rbf):
    at1 = np.array(Y1)
    ap1 = np.array(y1_rbf)
    return math.sqrt(((at1-ap1)**2).mean())/math.log(10)

def select_K(tup):
    li = list(tup)
    for i in range(3):
        del(li[2])
    for i in range(2):
        del(li[4])
    for i in (2,3,4):
        for j in range(-4,5):
            li.append(powermean(tup[8],tup[i],j))
        li.append(std(tup[8],tup[i])**2)
    del(li[3])
    return li
    #return (tup[0],tup[1],powermean(tup[8],tup[4],-1),std(tup[8],tup[4])**2,powermean(tup[8],tup[2],1),std(tup[8],tup[2]),tup[5],tup[9])
    #return tup


with open('./elasticity','rb') as f:
    data = pickle.load(f)
'''
data = []
dirs = os.listdir('./elas_data')
for i in dirs:
    with open('./elas_data/{:s}/elasticity'.format(i),'rb') as f:
        data.append(pickle.load(f))
'''

X = []
Y = []
for i in data:
    try:
        X.append(select_K(i[0]))
    except:
        continue
    Y.append(i[1])
sca = StandardScaler()
X = sca.fit_transform(X)

'''
#plotting validation curve
#param_range = [10,20,50,100,200,500,1000]
param_range = [1,2,5,10,20]
train_scores, test_scores = validation_curve(SVR(kernel='rbf',gamma=0.1,epsilon=0.1),X,Y,param_name='C',param_range=param_range,cv=10)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure()
plt.xlabel(r'$\alpha$')
plt.ylabel('Score')
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
'''



#plotting learning curve
train_sizes, train_scores, test_scores = learning_curve(
    SVR(kernel='rbf',C=10,gamma=0.1,epsilon=0.1),
    X,
    Y,
    train_sizes=[300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600],
    cv=10
)
plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
plt.legend(loc="best")
plt.show()
print(train_scores_mean)
print(test_scores_mean)


'''
#gridsearch
param = {
    'C':[1,2,5,10,20,50],
    'gamma':np.logspace(-5,0,6),
    'epsilon':np.logspace(-5,0,6)}
svr = SVR(kernel='rbf')
clf = GridSearchCV(svr,param)
clf.fit(X,Y)

for i in clf.cv_results_.keys():
    print(i,clf.cv_results_[i])
'''