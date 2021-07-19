import numpy as np
from tqdm import tqdm
import scipy
from scipy import linalg, optimize
import pandas as pd
import seaborn as sns
import xlrd
import matplotlib.pyplot as plt
import seaborn
import math
import sklearn
from sklearn import datasets
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
import scipy
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def split(data):
    test_ind = []
    train_ind = np.random.choice(data.index, size=int(0.66 * len(data.index)), replace=False)
    for i in range(len(data.index)):
        if not (data.index[i] in train_ind):
            test_ind.append(data.index[i])
    ind = data.index.isin(train_ind)
    train = data[ind]
    ind = data.index.isin(test_ind)
    test = data[ind]
    return train, test

def test_regression(type, data_not_data, target, data_name, target_name):
    data = pd.DataFrame(data_not_data)
    if type == 'numpy.float64' or max(target) > 10:
        reg = LinearRegression()
        model = reg.fit(data, target)
        model_pred = model.predict(data)
        if metrics.mean_squared_error(target, model_pred) < 10:
            print('///////////////////////////')
            print(data_name)
            print(target_name)
            print('///////////////////////////')
    if type == 'numpy.int64' and target.max <= 10:
        reg = LogisticRegression()
        model = reg.fit(data, target)
        model_pred = model.predict(data)
        if metrics.f1_score(target, model_pred, average='weighted') > 0.5:
            print(data_name)
            print(target_name)
            print('///////////////////////////')
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


heart=pd.read_excel(r"C:\Users\User\Documents\ML\heart.xls", "heart")
print(heart)
n = len(heart.index)
heart_0 = heart[heart['output'] == 0]
heart_1 = heart[heart['output'] == 1]


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# print('class 1 : {:.2f}\nclass 2 : {:.2f}'.format(np.sum(heart.output==0)/float(len(heart.output)),np.sum(heart.output==1)/float(len(heart.output))))
# for i in heart.columns:
#     for j in heart.columns:
#         if i != j:
#             test_regression(type(heart[j][0]),heart[i], heart[j], i, j)
#             if scipy.stats.spearmanr(heart[i], heart[j])[0] > 0.4 or scipy.stats.pearsonr(heart[i], heart[j])[0] > 0.4:
#                 print('///////////////////////')
#                 print(i)
#                 print(j)
#                 print(scipy.stats.spearmanr(heart[i], heart[j]))
#                 print(scipy.stats.pearsonr(heart[i], heart[j]))
#                 print('///////////////////////')
#
# corr = heart.corr()
# co = sns.heatmap(corr)
# plt.show()
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#попытка в кросс-валидацию
max = 0
for i in tqdm(range(100)):
    train_0, test_0 = split(heart_0)
    train_1, test_1 = split(heart_1)
    train = pd.concat([train_0, train_1], ignore_index=True)
    test = pd.concat([test_0, test_1], ignore_index=True)
    train_data = train.drop('output', axis = 1)
    train_target = train['output']
    test_data = test.drop('output', axis=1)
    test_target = test['output']
    rf = RandomForestClassifier(n_estimators=200, random_state=3, max_depth=2)
    model = rf.fit(train_data, train_target)
    model_pred = model.predict(test_data)
    if max < metrics.f1_score(test_target,model_pred, average='weighted'):
        max = metrics.f1_score(test_target,model_pred, average='weighted')
    print(metrics.f1_score(test_target,model_pred, average=None))
    print(metrics.f1_score(test_target,model_pred, average='macro'))
    print(metrics.f1_score(test_target,model_pred, average='micro'))
    print(metrics.f1_score(test_target,model_pred, average='weighted'))

print('////////////////////////////////////////////////')
print(max)
