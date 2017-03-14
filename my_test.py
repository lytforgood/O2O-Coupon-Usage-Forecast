# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
###########
data_all = pd.read_csv('data/train01.csv')
from sklearn.model_selection import train_test_split
X=data_all.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
y=data_all.label
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=0)##test_size测试集合所占比例
test_preds = pd.DataFrame({"label":test_y})

###########
train = pd.read_csv('data/train02.csv')
test = pd.read_csv('data/train03.csv')

train_y = train.label
train_x = train.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
test_y = test.label
test_x = test.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)

# test_preds = test[['user_id','label']]
test_preds = pd.DataFrame({"label":test_y})
###########


train_x = xgb.DMatrix(train_x,label=train_y)
test_x = xgb.DMatrix(test_x)

params={'booster':'gbtree',
      'objective': 'rank:pairwise', ##一种排序算法
      'eval_metric':'auc',
      'gamma':0.1,
      'min_child_weight':1.1,
      'max_depth':5,
      'lambda':10,
      'subsample':0.7,
      'colsample_bytree':0.7,
      'colsample_bylevel':0.7,
      'eta': 0.01,
      'tree_method':'exact',
      'seed':0,
      'nthread':12
      }

#train on dataset1, evaluate on dataset2
#watchlist = [(dataset1,'train'),(dataset2,'val')]
#model = xgb.train(params,dataset1,num_boost_round=3000,evals=watchlist,early_stopping_rounds=300)

watchlist = [(train_x,'train')]
model = xgb.train(params,train_x,num_boost_round=100,evals=watchlist)
#predict test set
test_preds['pre_proba'] = model.predict(test_x)  ##-1.0081545~2.5258696 dataframe可以直接赋值一列
test_preds.pre_proba = MinMaxScaler().fit_transform(test_preds.pre_proba) ##规范化到0~1

test_preds['prelabel']=test_preds.pre_proba.apply(lambda x:1 if x>0.5 else 0)
# test_preds.sort_values(by=['user_id','label'],inplace=True)

from sklearn import metrics
print "AUC Score: %f" % metrics.roc_auc_score(test_preds['label'], test_preds['pre_proba'])
print"Accuracy : %.4g" % metrics.accuracy_score(test_preds['label'], test_preds['prelabel'])



import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from  sklearn.ensemble  import  GradientBoostingClassifier
###########
data_all = pd.read_csv('data/train01.csv')

data_all=data_all.fillna(0)

from sklearn.model_selection import train_test_split
X=data_all.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
y=data_all.label
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=0)##test_size测试集合所占比例
test_preds = pd.DataFrame({"label":test_y})

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
).fit(train_x, train_y)  ##多类别回归建议使用随机森林

test_preds['prelabel'] = clf.predict(test_x)
test_preds['pre_proba']=clf.predict_proba(test_x)[:,1]

from sklearn import metrics
print "AUC Score: %f" % metrics.roc_auc_score(test_preds['label'], test_preds['pre_proba'])
print"Accuracy : %.4g" % metrics.accuracy_score(test_preds['label'], test_preds['prelabel'])

