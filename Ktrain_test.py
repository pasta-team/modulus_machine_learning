#!/home/jjs/miniconda3/bin/python

import os
import pickle,linecache,math
import numpy as np
from pasta.element import Element
from sklearn.svm import SVR
from sklearn import linear_model,metrics
from math import sqrt
#import optunity
#import optunity.metrics
#from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from random import shuffle
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
        for j in range(0,1):
            li.append(powermean(tup[8],tup[i],j))
        li.append(std(tup[8],tup[i])**2)
    return li
    #return (tup[0],tup[1],powermean(tup[8],tup[4],-1),std(tup[8],tup[4])**2,powermean(tup[8],tup[2],1),std(tup[8],tup[2]),tup[5],tup[9])
    #return tup

def select_G(tup):
    return (tup[0],tup[1],powermean(tup[8],tup[4],1),std(tup[8],tup[4]),tup[5])
    #return tup


with open('./symelasticity','rb') as f:
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
shuffle(data)
C = 10
gamma = 0.1
epsilon = 0.1

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

print(C,gamma,epsilon)
for fit in range(10):
    test_data = data[fit*test_len:min((fit+1)*test_len,total_len)]
    if fit == 0:
        train_data = data[test_len:]
    elif fit == 9:
        train_data = data[:9*test_len]
    else:
        train_data = data[:fit*test_len]+data[(fit+1)*test_len:]
    #X1 = []
    X2 = []
    #Y1 = []
    Y2 = []
    for i in train_data:
        #X1.append(select_K(data[i][0]))
        #Y1.append(data[i][1])
        try:
            X2.append(select_K(i[0]))
        except:
            #print(data[i])
            continue
        Y2.append(i[1])
    #print(len(X2))
    #ss1 = StandardScaler()
    ss2 = StandardScaler()
    #X1 = ss1.fit_transform(X1)
    X2 = ss2.fit_transform(X2)
    #resx1 = []
    resx2 = []
    #resy1 = []
    resy2 = []
    for i in test_data:
        #resx1.append(select_K(i[0]))
        #resy1.append(i[1])
        try:
            resx2.append(select_K(i[0]))
        except:
            continue
        resy2.append(i[1])
    #print(len(resx2))
    #resx1 = ss1.transform(resx1)
    resx2 = ss2.transform(resx2)
    #svr1 = SVR(kernel='rbf',C=C,gamma=gamma,epsilon=epsilon)
    svr2 = SVR(kernel='rbf',C=C,gamma=gamma,epsilon=epsilon)
    #l1 = svr1.fit(X1,Y1)
    #y1_rbf = l1.predict(X1)
    #prey1 = l1.predict(resx1)
    l2 = svr2.fit(X2,Y2)
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
    '''
    reg1 = linear_model.LinearRegression()
    ay1 = y2_rbf.reshape(-1,1)
    reg1.fit(ay1,Y2)
    print(reg1.score(ay1,Y2))
    reg2 = linear_model.LinearRegression()
    ay2 = prey2.reshape(-1,1)
    reg2.fit(ay2,resy2)
    print(reg2.score(ay2,resy2))
    '''
'''
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
'''

print((np.mean(r5),np.mean(r10),np.mean(r20),np.mean(r30)))
print((np.mean(e5),np.mean(e10),np.mean(e20),np.mean(e30)))

#reg = linear_model.LinearRegression()
fix = np.array([math.exp(i) for i in lifix]).reshape(-1,1)
fiy = np.array([math.exp(i) for i in lifiy]).reshape(-1,1)
print('r2 : {:.4g}'.format(metrics.r2_score(fix,fiy)))
print('RMSE : {:.4g}'.format(sqrt(metrics.mean_squared_error(fix,fiy))))
print('MAE : {:.4g}'.format(metrics.mean_absolute_error(fix,fiy)))
#reg.fit(fix,fiy)
#print(reg.score(fix,fiy))
'''
    plt.figure(1)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    plt.sca(ax1)
    plt.scatter(Y1,y1_rbf,color = 'darkorange',label='K_train',s=1)
    plt.plot([0,6],[0,6],color = 'black')
    plt.legend()
    plt.sca(ax2)
    plt.scatter(Y2,y2_rbf,color = 'darkorange',label='G_train',s=1)
    plt.legend()
    plt.plot([0,6],[0,6],color = 'black')
    plt.sca(ax3)
    plt.scatter(resy1,prey1,color = 'darkorange',label='K_test',s=1)
    plt.legend()
    plt.plot([0,6],[0,6],color = 'black')
    plt.sca(ax4)
    plt.scatter(resy2,prey2,color = 'darkorange',label='G_test',s=1)
    plt.plot([0,6],[0,6],color = 'black')
    plt.legend()
    plt.show()
'''
'''
parameters = {'C':[5e1,1e2,2e2,5e2,1e3,2e3],
              'epsilon':[0.01,0.02,0.05,0.1,0.2,0.5,1]}

gs = GridSearchCV(SVR(),parameters,cv=10,verbose=2,refit=True)
%time_ = gs.fit(X1,Y1)
print(gs.best_params_,gs.best_score_)
print(gs.score(resx1,resy1))

@optunity.cross_validated(x=X2, y=Y2, num_folds=10, num_iter=2)
def svm_mse(x_train, y_train, x_test, y_test, C, gamma):
    model = SVR(C=C, gamma=gamma).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return optunity.metrics.mse(y_test, y_pred)
optimal_pars, _, _ = optunity.minimize(svm_mse, num_evals=200, C=[0,200], gamma=[0,1])
optimal_model = SVR(**optimal_pars).fit(X2,Y2)
y2_rbf = optimal_model.predict(X2)
#prey2 = optimal_model.predict(resx2)
'''
'''
print(**optimal_pars)
print(mse(Y2,y2_rbf))
print(watch(Y2,y2_rbf))
#print(mse(resy2,prey2))
#print(watch(resy2,prey2))

def para(C,gamma):
    svr_rbf = SVR(kernel='rbf',C=C,gamma=gamma)
    l1 = svr_rbf.fit(X1,Y1)
    y1_rbf = l1.predict(X1)
    l2 = svr_rbf.fit(X2,Y2)
    y2_rbf = l2.predict(X2)
    watch(Y1,Y2,y1_rbf,y2_rbf)
    plt.figure()
    plt.scatter(Y1,y1_rbf,color = 'darkorange',label='K')
    plt.figure()
    plt.scatter(Y2,y2_rbf,color = 'darkorange',label='G')
'''