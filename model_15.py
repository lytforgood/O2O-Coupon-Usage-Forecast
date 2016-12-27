# -*- coding: utf-8 -*-
"""
#简化版模型运行
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVR  特征做细做精 不平衡数据的降采样
#读入特征--样本  21列 inf  修改  离散化(eg:1-100离散到10个区间kmeans聚类) 交叉特征(按照特征非0出现比率删除最少得到那部分特征) onehot编码 1基础算法2特殊情况
# train_sample=train_sample_allend
# test_sample=test_sample_allend
# test_data=test_pre_allend_all
# test_pre_allend_1 = pd.read_csv("../../ccf_data/result/test_pre_allend_1.csv",header=None)
test_data = pd.read_csv("../../ccf_data/result/test_pre_allend_all.csv",header=None)
train_sample = pd.read_csv("../../ccf_data/result/train_sample_allend.csv",header=None)
test_sample = pd.read_csv("../../ccf_data/result/test_sample_allend.csv",header=None)
train_all=pd.concat([test_sample,train_sample]) #行合并
#选择模型
clf = LogisticRegression(penalty='l2', C=2)
#clf = RandomForestClassifier(n_estimators =50)
# clf = GradientBoostingClassifier(n_estimators = 200)
# clf = SVC(C = 0.8,probability = True)
#进行训练   train_sample.iloc[:,7:16].dtypes 查看数据类型
clf.fit(train_sample.iloc[:,7:29], train_sample.iloc[:,29])
#clf.fit(train_all.iloc[:,7:25], train_all.iloc[:,25])
#预测概率
# test_sample_feature_proba = clf.predict_proba(test_sample.iloc[:,7:25])
test_sample_feature_proba = clf.predict_proba(test_sample.iloc[:,7:29])
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(list(test_sample.iloc[:,29]), test_sample_feature_proba[:,1])
roc_auc = auc(fpr, tpr)
print 'roc_auc:%f' %roc_auc
#按照cid计算单独auc然后求auc均值
t1=pd.Series(test_sample_feature_proba[:,1])
test_data_auc=pd.concat([test_sample.iloc[:,[0,1,29]],t1],axis=1)
data = test_data_auc.iloc[:,1]
data = list(data.drop_duplicates())  #去重
count = 0
sum_roc_auc = 0.0
for i in range(len(data)):
    tmp=test_data_auc[test_data_auc.iloc[:,1]==data[0]]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(tmp.iloc[:,2], tmp.iloc[:,3])
    roc_auc = auc(fpr, tpr)
    sum_roc_auc = sum_roc_auc + roc_auc
    count = count + 1
print 'avg_auc:%f' %(sum_roc_auc/count)
#预测最后结果
clf = LogisticRegression(penalty='l2', C=2)
#clf = RandomForestClassifier(n_estimators =50)
# clf = GradientBoostingClassifier(n_estimators = 200)
# clf = SVC(C = 0.8,probability = True)
#训练
clf.fit(train_sample.iloc[:,7:16], train_sample.iloc[:,16])
# clf.fit(train_all.iloc[:,7:16], train_all.iloc[:,16])
test_sample_feature_proba = clf.predict_proba(test_data.iloc[:,6:28])
#将预测结果转换成DF
t1=pd.Series(test_sample_feature_proba[:,1])
index=[]
for i in range(len(t1)):
    index.append(i)
test_data.index=index
rex=pd.concat([test_data.iloc[:,[0,2,5]],t1],axis=1)  #追加列
rex2=test_pre_allend_1.iloc[:,[0,2,5,6]]
rex.columns=['c0','c2','c5','lab']
rex2.columns=['c0','c2','c5','lab']
rex3=pd.concat([rex,rex2])
rex3=rex3.drop_duplicates(['c0','c2','c5','lab'])
rex3.iloc[:,[0,1,2]].astype(int).astype(str)
rex3.to_csv("../../ccf_data/result/sample_submission1.csv",header=False,index=False)


# #回归方法Regressor
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor
# from xgboost import XGBClassifier
# seed=0
# # clf = Ridge(alpha=1.0,random_state=seed) #0.841269 0.732074
# # clf = Lasso(alpha=0.001,random_state=seed) #0.834484 0.716615
# # clf = ElasticNet(alpha=0.001,random_state=seed) #0.829437 0.716322
# # clf = DecisionTreeRegressor(max_depth=5,random_state=seed) #0.635430 0.531403
# # clf = ExtraTreesRegressor(n_jobs=-1,n_estimators=100,random_state=seed) #0.649415 0.544799
# # clf = AdaBoostRegressor(n_estimators=100,random_state=seed) #0.635245 0.531407
# # clf = GradientBoostingRegressor(n_estimators=50,random_state=seed) #0.562742 0.474085
# # clf = XGBRegressor(n_estimators=1000,seed=seed) #0.718094 0.558765
# # clf = XGBClassifier(n_estimators=1000,seed=seed)  #0.672249 0.588265
# #进行训练
# clf.fit(train_sample.iloc[:,7:25], train_sample.iloc[:,25])
# #clf.fit(train_all.iloc[:,7:25], train_all.iloc[:,25])
# #预测概率
# # test_sample_feature_proba = clf.predict_proba(test_sample.iloc[:,7:25])
# test_sample_feature_proba = clf.predict(test_sample.iloc[:,7:25])
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# fpr, tpr, _ = roc_curve(list(test_sample.iloc[:,25]), test_sample_feature_proba)
# roc_auc = auc(fpr, tpr)
# print 'roc_auc:%f' %roc_auc
# #按照cid计算单独auc然后求auc均值
# t1=pandas.Series(test_sample_feature_proba)
# test_data_auc=pandas.concat([test_sample.iloc[:,[0,1,25]],t1],axis=1)
# data = test_data_auc.iloc[:,1]
# data = list(data.drop_duplicates())  #去重
# count = 0
# sum_roc_auc = 0.0
# for i in range(len(data)):
#     tmp=test_data_auc[test_data_auc.iloc[:,1]==data[0]]
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     fpr, tpr, _ = roc_curve(tmp.iloc[:,2], tmp.iloc[:,3])
#     roc_auc = auc(fpr, tpr)
#     sum_roc_auc = sum_roc_auc + roc_auc
#     count = count + 1
# print 'avg_auc:%f' %(sum_roc_auc/count)
# clf.fit(train_all.iloc[:,7:25], train_all.iloc[:,25])
# test_sample_feature_proba = clf.predict(test_data.iloc[:,6:24])
# #将预测结果转换成DF
# t1=pandas.Series(test_sample_feature_proba)
# rex=pandas.concat([test_data.iloc[:,[0,2,5]],t1],axis=1)  #追加列
# rex.iloc[:,[0,1,2]].astype(int).astype(str)
# rex.to_csv("../../ccf_data/result/sample_submission1.csv",header=False,index=False)
