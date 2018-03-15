#!/home/jjs/miniconda3/bin/python

import os
import pickle,linecache,math
import numpy as np
from pasta.element import Element
import pandas as pd
from sklearn import cross_validation, metrics
from sklearn.ensemble import BaggingRegressor
from sklearn.grid_search import GridSearchCV
from random import shuffle
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

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
    #res1 = (at1-ap1)**2
    #res2 = (at2-ap2)**2
    #pos = np.argwhere(res1 == max(res1))[0][0]
    #res1.sort()
    #res2.sort()
    #return(Y1[pos],res1[-10:],res2[-10:])
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

def select_G(tup):
    return (tup[0],tup[1],powermean(tup[8],tup[4],1),std(tup[8],tup[4]),tup[5])
    #return tup


with open('./elasticity','rb') as f:
    data = pickle.load(f)
'''
data = []
f = open('./rdfelasticity','rb')
for i in range(4089):
    data.append(pickle.load(f))
f.close()
'''

total_len = len(data)
test_len = total_len//10+1
train_len = total_len - test_len
shuffle(data)

bagging = BaggingRegressor(max_samples=0.5,max_features=0.5,oob_score=False)

m1 = []
r5 = []
r10 = []
r20 = []
r30 = []
m2 = []
e5 = []
e10 = []
e20 = []
e30 = []
lifix = []
lifiy = []

gm1 = []
gr5 = []
gr10 = []
gr20 = []
gr30 = []
gm2 = []
ge5 = []
ge10 = []
ge20 = []
ge30 = []
glifix = []
glifiy = []

for fit in range(10):
    test_data = data[fit*test_len:min((fit+1)*test_len,total_len)]
    if fit == 0:
        train_data = data[test_len:]
    elif fit == 9:
        train_data = data[:9*test_len]
    else:
        train_data = data[:fit*test_len]+data[(fit+1)*test_len:]
    X2 = []
    Y2 = []
    for i in train_data:
        try:
            X2.append(select_K(i[0]))
        except:
            continue
        Y2.append(i[1])
    ss2 = StandardScaler()
    X2 = ss2.fit_transform(X2)
    resx2 = []
    resy2 = []
    for i in test_data:
        try:
            resx2.append(select_K(i[0]))
        except:
            continue
        resy2.append(i[1])
    resx2 = ss2.transform(resx2)


    l2 = bagging.fit(X2,Y2)
    y2_rbf = l2.predict(X2)
    prey2 = l2.predict(resx2)
    m1.append(rmse(Y2,y2_rbf))
    actr = watch(Y2,y2_rbf)
    r5.append(actr[0])
    r10.append(actr[1])
    r20.append(actr[2])
    r30.append(actr[3])
    m2.append(rmse(resy2,prey2))
    acte = watch(resy2,prey2)
    e5.append(acte[0])
    e10.append(acte[1])
    e20.append(acte[2])
    e30.append(acte[3])
    lifix.extend(prey2)
    lifiy.extend(resy2)

    reg2 = linear_model.LinearRegression()
    ay2 = prey2.reshape(-1,1)
    reg2.fit(ay2,resy2)
    print(reg2.score(ay2,resy2))

print(np.mean(m1),np.std(m1))
print(np.mean(r5),np.std(r5))
print(np.mean(r10),np.std(r10))
print(np.mean(r20),np.std(r20))
print(np.mean(r30),np.std(r30))
print(np.mean(m2),np.std(m2))
print(np.mean(e5),np.std(e5))
print(np.mean(e10),np.std(e10))
print(np.mean(e20),np.std(e20))
print(np.mean(e30),np.std(e30))

reg = linear_model.LinearRegression()
fix = np.array(lifix).reshape(-1,1)
fiy = np.array(lifiy).reshape(-1,1)
reg.fit(fix,fiy)
print(reg.score(fix,fiy))
