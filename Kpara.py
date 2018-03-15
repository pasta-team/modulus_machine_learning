#!/home/jjs/miniconda3/bin/python

import os
import pickle,linecache,math
import numpy as np
from pasta.element import Element
from sklearn.svm import SVR
#import optunity
#import optunity.metrics
#from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from random import shuffle
import matplotlib.pyplot as plt

def powermean(w,x,n):
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
    return (tup[0],tup[1],powermean(tup[8],tup[4],1),std(tup[8],tup[4])**2,powermean(tup[8],tup[2],1),std(tup[8],tup[2]),tup[5])
    #return tup

def select_G(tup):
    return (tup[0],tup[1],powermean(tup[8],tup[4],1),std(tup[8],tup[4]),tup[5])
    #return tup

with open('./elasticity','rb') as f:
    data = pickle.load(f)

total_len = len(data)
test_len = total_len//10+1
dic = {}
ave = {}
#C = 50
#gamma = 0.1
#epsilon = 0.1

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

for ite in range(10):
    shuffle(data)
    for C in (20,50,100,200):
        for gamma in (0.05,0.1,0.2):
            for epsilon in (0.05,0.1,0.2):
                print(C,gamma,epsilon)
                for fit in range(10):
                    test_data = data[fit*test_len:min((fit+1)*test_len,total_len)]
                    if fit == 0:
                        train_data = data[test_len:]
                    elif fit == 9:
                        train_data = data[9*test_len:]
                    else:
                        train_data = data[:fit*test_len]+data[(fit+1)*test_len:]
                    X1 = []
                    #X2 = []
                    Y1 = []
                    #Y2 = []
                    for i in train_data:
                        #Y1.append(data[i][1])
                        try:
                            X1.append(select_K(i[0]))
                        except:
                            #print(data[i])
                            continue
                        Y1.append(i[1])
                    #ss1 = StandardScaler()
                    ss1 = StandardScaler()
                    #X1 = ss1.fit_transform(X1)
                    X1 = ss1.fit_transform(X1)
                    #resx1 = []
                    resx1 = []
                    #resy1 = []
                    resy1 = []
                    for i in test_data:
                        #resx1.append(select_K(i[0]))
                        #resy1.append(i[1])
                        try:
                            resx1.append(select_K(i[0]))
                        except:
                            continue
                        resy1.append(i[1])
                    #resx1 = ss1.transform(resx1)
                    resx1 = ss1.transform(resx1)
                    #svr1 = SVR(kernel='rbf',C=C,gamma=gamma,epsilon=epsilon)
                    svr1 = SVR(kernel='rbf',C=C,gamma=gamma,epsilon=epsilon)
                    #l1 = svr1.fit(X1,Y1)
                    #y1_rbf = l1.predict(X1)
                    #prey1 = l1.predict(resx1)
                    l1 = svr1.fit(X1,Y1)
                    y1_rbf = l1.predict(X1)
                    prey1 = l1.predict(resx1)
                    if (C,gamma,epsilon) in dic:
                        dic[(C,gamma,epsilon)].append(rmse(resy1,prey1))
                    else:
                        dic[(C,gamma,epsilon)] = [rmse(resy1,prey1)]
                    '''
                    m1.append(rmse(Y1,y1_rbf))
                    actr = watch(Y1,y1_rbf)
                    r5.append(actr[0])
                    r10.append(actr[1])
                    r20.append(actr[2])
                    r30.append(actr[3])
                    m2.append(rmse(resy1,prey1))
                    acte = watch(resy1,prey1)
                    e5.append(acte[0])
                    e10.append(acte[1])
                    e20.append(acte[2])
                    e30.append(acte[3])
                
                res = np.mean(m2)+np.std(m2)
                print(res)
                if (C,gamma,epsilon) in dic:
                    dic[(C,gamma,epsilon)].append(res)
                else:
                    dic[(C,gamma,epsilon)]=[res]
                    '''

for i in dic:
    ave[i] = np.mean(dic[i])+np.std(dic[i])

print(dic)
print(ave)

#print(np.mean(m1),np.std(m1))
#print(np.mean(r5),np.std(r5))
#print(np.mean(r10),np.std(r10))
#print(np.mean(r20),np.std(r20))
#print(np.mean(r30),np.std(r30))
#print(np.mean(m2),np.std(m2))
#print(np.mean(e5),np.std(e5))
#print(np.mean(e10),np.std(e10))
#print(np.mean(e20),np.std(e20))
#print(np.mean(e30),np.std(e30))
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