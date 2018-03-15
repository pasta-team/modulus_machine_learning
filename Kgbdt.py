#!/home/jjs/miniconda3/bin/python

import os
import pickle,linecache,math
import numpy as np
from pasta.element import Element
import xgboost as xgb
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from random import shuffle
from math import sqrt,log
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
    li = [tup[1]]
    for i in (2,3,4):
        li.append(powermean(tup[8],tup[i],-1))
        li.append(std(tup[8],tup[i]))
    li.append(np.mean(tup[10]))
    li.append(np.std(tup[10]))
    #li.append()
    return li
    # li = list(tup)
    # for i in range(3):
    #     del(li[2])
    # for i in range(2):
    #     del(li[4])
    # del(li[5])
    # for i in (2,3,4):
    #     for j in range(-4,5):
    #         li.append(powermean(tup[8],tup[i],j))
    #     li.append(std(tup[8],tup[i])**2)
    # li.append(np.mean(tup[9]))
    # li.append(np.std(tup[9]))
    # del(li[3])
    # return li
    #return (tup[0],tup[1],powermean(tup[8],tup[4],-1),std(tup[8],tup[4])**2,powermean(tup[8],tup[2],1),std(tup[8],tup[2]),tup[5],tup[9])
    #return tup

def select_G(tup):
    return (tup[0],tup[1],powermean(tup[8],tup[4],1),std(tup[8],tup[4]),tup[5])
    #return tup

# with open('./elasticity','rb') as f:
#     data = pickle.load(f)

data = []
dirs = os.listdir('./elas_data')
for i in dirs:
    try:
        with open('./elas_data/{:s}/elasticity'.format(i),'rb') as f:
            data.append(pickle.load(f))
    except:
        continue


total_len = len(data)
test_len = total_len//10+1
train_len = total_len - test_len
shuffle(data)

ptrain = []
ptest = []
ltrain = []
ltest = []
for i in range(train_len):
    try:
        ptrain.append(select_K(data[i][0]))
    except:
        continue
    ltrain.append(data[i][1])
for i in range(train_len,total_len):
    try:
        ptest.append(select_K(data[i][0]))
    except:
        continue
    ltest.append(data[i][1])

'''
print(type(ptrain[0]))
print(len(ptrain[0]))
print(len(ptrain))
print(type(ptrain))
'''
ptrain = np.array(ptrain)
ltrain = np.array(ltrain)
ptest = np.array(ptest)
ltest = np.array(ltest)
'''
print(type(ptrain[0]))
print(type(ptrain))
'''
print(ptrain.shape)
print(ltrain.shape)
print(ptest.shape)
print(ltest.shape)



'''
param = {'bst:max_depth':2,'bst:eta':0.3,'silent':1}
plst = param.items()
num_round = 100
#bst = xgb.cv(param,dtrain,num_round,nfold=5,seed=0,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
evallist = [(dtest,'eval'),(dtrain,'train')]
gboos = xgb.Booster(param)
bst = gboos.train(dtrain,num_round,evallist)
#bst = xgb.train(plst,dtrain,num_round,evallist)
#bst.save_model('0001.model')
print(bst.get_fscore())
'''


def modelfit(alg, ptrain, ltrain, ptest, ltest, useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(ptrain, label=ltrain)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, early_stopping_rounds=early_stopping_rounds,)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(ptrain, ltrain)

    #Predict training set:
    dtrain_predictions = alg.predict(ptrain)
    dtest_predictions = alg.predict(ptest)
    #dtrain_predprob = alg.predict_proba(ptrain)[:,1]

    #Print model report:
    print('\nModel Report')
    tr1 = [math.exp(i) for i in ltrain]
    tr2 = [math.exp(i) for i in dtrain_predictions]
    te1 = [math.exp(i) for i in ltest]
    te2 = [math.exp(i) for i in dtest_predictions]
    #print('r2 : {:.4g}'.format(metrics.r2_score(ltrain, dtrain_predictions)))
    #print('MSE : {:.4g}'.format(metrics.mean_squared_error(ltrain, dtrain_predictions)))
    #print('r2 : {:.4g}'.format(metrics.r2_score(ltest, dtest_predictions)))
    #print('MSE : {:.4g}'.format(metrics.mean_squared_error(ltest, dtest_predictions)))
    print('r2 : {:.4g}'.format(metrics.r2_score(tr1,tr2)))
    print('RMSE : {:.4g}'.format(sqrt(metrics.mean_squared_error(tr1,tr2))))
    print('MAE : {:.4g}'.format(metrics.mean_absolute_error(tr1,tr2)))
    print('r2 : {:.4g}'.format(metrics.r2_score(te1,te2)))
    print('RMSE : {:.4g}'.format(sqrt(metrics.mean_squared_error(te1,te2))))
    print('MAE : {:.4g}'.format(metrics.mean_absolute_error(te1,te2)))
    print(watch(ltrain,dtrain_predictions))
    print(watch(ltest,dtest_predictions))
    #print('AUC Score (Train): {:f}'.format(metrics.roc_auc_score(ltrain, dtrain_predprob)))


    ax = xgb.plot_importance(alg)
    ax.plot()
    plt.show()

'''
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
'''

'''
param_test1 = {
    #'max_depth':[11,12,13,14,15],
    #'min_child_weight':[1,2,3,4,5],
    #'gamma':[i/10.0 for i in range(0,5)]
    #'subsample':[i/10.0 for i in range(6,10)],
    #'colsample_bytree':[i/10.0 for i in range(6,10)],
    #'reg_alpha':[1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,0.1,0.2,0.5],
    'learning_rate':[0.01,0.02,0.05,0.1,0.2,0.5],
}
gsearch1 = GridSearchCV(
    estimator = XGBRegressor(
        subsample=0.6,
        colsample_bytree=0.8,
        min_child_weight=3,
        max_depth=15,
        gamma=0
        ),
    param_grid = param_test1,
    cv = 10)
gsearch1.fit(ptrain,ltrain)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

'''
xgb1 = XGBRegressor(
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    max_depth=15,
    gamma=0,
    reg_alpha=0.01)
modelfit(xgb1,ptrain,ltrain,ptest,ltest)




'''
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

    l1 = gbdt.fit(X2,Y2)
    y2_rbf = l1.predict(X2)
    prey2 = l1.predict(resx2)
    gm1.append(rmse(Y2,y2_rbf))
    actr = watch(Y2,y2_rbf)
    gr5.append(actr[0])
    gr10.append(actr[1])
    gr20.append(actr[2])
    gr30.append(actr[3])
    gm2.append(rmse(resy2,prey2))
    acte = watch(resy2,prey2)
    ge5.append(acte[0])
    ge10.append(acte[1])
    ge20.append(acte[2])
    ge30.append(acte[3])
    glifix.extend(prey2)
    glifiy.extend(resy2)


    reg1 = linear_model.LinearRegression()
    ay1 = y2_rbf.reshape(-1,1)
    reg1.fit(ay1,Y2)
    print(reg1.score(ay1,Y2))
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

print(np.mean(gm1),np.std(gm1))
print(np.mean(gr5),np.std(gr5))
print(np.mean(gr10),np.std(gr10))
print(np.mean(gr20),np.std(gr20))
print(np.mean(gr30),np.std(gr30))
print(np.mean(gm2),np.std(gm2))
print(np.mean(ge5),np.std(ge5))
print(np.mean(ge10),np.std(ge10))
print(np.mean(ge20),np.std(ge20))
print(np.mean(ge30),np.std(ge30))

reg = linear_model.LinearRegression()
fix = np.array(lifix).reshape(-1,1)
fiy = np.array(lifiy).reshape(-1,1)
reg.fit(fix,fiy)
print(reg.score(fix,fiy))

gfix = np.array(glifix).reshape(-1,1)
gfiy = np.array(glifiy).reshape(-1,1)
reg.fit(gfix,gfiy)
print(reg.score(gfix,gfiy))
'''