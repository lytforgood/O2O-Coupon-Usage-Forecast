# -*- coding: utf-8 -*-
"""
#xgb  gbdt
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import metrics

test_data = pd.read_csv("/Users/yuyin/Downloads/笔记学习/天池比赛/o2o优惠券预测/第一赛季/O2O优惠券使用预测/ccf_data/result/test_pre_allend_all.csv",header=None)
train_sample = pd.read_csv("/Users/yuyin/Downloads/笔记学习/天池比赛/o2o优惠券预测/第一赛季/O2O优惠券使用预测/ccf_data/result/train_sample_allend.csv",header=None)
test_sample = pd.read_csv("/Users/yuyin/Downloads/笔记学习/天池比赛/o2o优惠券预测/第一赛季/O2O优惠券使用预测/ccf_data/result/test_sample_allend.csv",header=None)
train_all=pd.concat([test_sample,train_sample]) #行合并

X=train_all.iloc[:,7:29]
y=train_all.iloc[:,29]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=0)##test_size测试集合所占比例

train_x, train_y =train_sample.iloc[:,7:29],train_sample.iloc[:,29]
test_x, test_y =test_sample.iloc[:,7:29],test_sample.iloc[:,29]

test_preds = pd.DataFrame({"label":test_y})

##GBDT
clf = GradientBoostingClassifier(
loss='deviance',  ##损失函数默认deviance  deviance具有概率输出的分类的偏差
n_estimators=100, ##默认100 回归树个数 弱学习器个数
learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
max_depth=3,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
subsample=1,  ##树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
min_samples_split=2, ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
min_samples_leaf=1, ##叶节点所需的最小样本数  如果是浮点数代表是百分比
max_features=None, ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
max_leaf_nodes=None, ##叶节点的数量 None不限数量
min_impurity_split=1e-7, ##停止分裂叶子节点的阈值
verbose=0,  ##打印输出 大于1打印每棵树的进度和性能
warm_start=False, ##True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
random_state=0  ##随机种子-方便重现
)
clf.fit(train_x,train_y)
##xgb
clf = XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#nthread=4,# cpu 线程数 默认最大
learning_rate= 0.3, # 如同学习率
min_child_weight=1,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=6, # 构建树的深度，越大越容易过拟合
gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=1, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样
reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#reg_alpha=0, # L1 正则项参数
#scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
#objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
#num_class=10, # 类别数，多分类与 multisoftmax 并用
n_estimators=100, #树的个数
seed=1000 #随机种子
#eval_metric= 'auc'
)
clf.fit(train_x,train_y,eval_metric='auc')

##网格搜索
from sklearn.model_selection import GridSearchCV
tuned_parameters= [{'n_estimators':range(20,81,10),
                  'max_depth':range(3,14,2),
                  'learning_rate':[0.1, 0.5, 1.0],
                  'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                  }]
clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5,
                       scoring='roc_auc')
# clf=GridSearchCV(
# estimator, ##模型
# param_grid, ##参数字典或者字典列表
# scoring=None,  ##评价分数的方法
# fit_params=None, ##fit的参数 字典
# n_jobs=1, ##并行数  -1全部启动
# iid=True,  ##每个cv集上等价
# refit=True,  ##使用整个数据集重新编制最佳估计量
# cv=None,   ##几折交叉验证None默认3
# verbose=0, ##控制详细程度：越高，消息越多
# pre_dispatch='2*n_jobs',  ##总作业的确切数量
# error_score='raise',  ##错误时选择的分数
# return_train_score=True   ##如果'False'，该cv_results_属性将不包括训练得分
# )
clf.fit(X_train, y_train)
print(clf.best_params_)



test_preds['prelabel'] = clf.predict(test_x)
test_preds['pre_proba']=clf.predict_proba(test_x)[:,1]


print "AUC Score: %f" % metrics.roc_auc_score(test_preds['label'], test_preds['pre_proba'])
print"Accuracy : %.4g" % metrics.accuracy_score(test_preds['label'], test_preds['prelabel'])


