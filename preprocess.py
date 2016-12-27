# -*- coding: utf-8 -*-
'''
数据预处理
---------
数据划分---预测数据7.1-7.31  预测未来15天内是否消费   用前三个月的统计量预测后15天的结果
训练数据---1.1--6.30
划分数据集合
训练集1.1~4.1--4.15-5.15
验证集2.1~5.1--5.15-6.15
测试集合3.15~6.15--7.1-7.30

#利用sql执行数据框
from pandasql import sqldf
pysqldf=lambda q:sqldf(q,globals())
pysqldf("select * from test_sample  limit 5")
'''
import pandas as pd
from pandasql import sqldf
import datetime

off_train = pd.read_csv("../original_data/ccf_offline_stage1_train.csv",header=None)
# on_train = pd.read_csv("../original_data/ccf_online_stage1_train.csv",header=None)
off_test = pd.read_csv("../original_data/ccf_offline_stage1_test.csv",header=None)

# del on_train
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# train_all=pandas.concat([test_sample,train_sample]) #行合并
off_train.columns=['c0','c1','c2','c3','c4','c5','c6']
# on_train.columns=['c0','c1','c2','c3','c4','c5','c6']
off_test.columns=['c0','c1','c2','c3','c4','c5']
#从领取时间计算  正样本1-负样本2-普通消费3
#正样本1  领取优惠券并且使用
model_off_train_1=off_train[(off_train.iloc[:,6]!="null")&(off_train.iloc[:,2]!="null")]
#负样本2  领取优惠券但未使用
model_off_train_2=off_train[(off_train.iloc[:,6]=="null")&(off_train.iloc[:,2]!="null")]
#普通消费3 未领取但购买  普通消费
model_off_train_3=off_train[(off_train.iloc[:,6]!="null")&(off_train.iloc[:,2]=="null")]


##1.1~4.1--4.15-5.15 #model_off_train_1.dtypes 查看所有类型
train_sample_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160515]
train_sample_2=model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160415]
train_sample_3=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160515]
train_sample_all=pd.concat([train_sample_1,train_sample_2,train_sample_3])#train_sample_all.shape
train_sample_all.columns=['c0','c1','c2','c3','c4','c5','c6']


#统计特征
#1.1~4.1期间  用户在该店铺领取的的优惠券个数
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160401],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160401]])
# feature_1.columns=['c0','c1','c2','c3','c4','c5','c6']
# feature_1=feature_1.groupby(['c0','c1'])['c2'].count() #sql代替group by
pysqldf=lambda q:sqldf(q,globals())
feature_1=pysqldf("select c0,c1,count(c2) as f1 from feature_1 group by c0,c1")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0','c1'])
#缺少值
# train_sample_all[pd.isnull(train_sample_all.iloc[:,7])].head()  #查看缺失值
train_sample_all.iloc[:,7]=train_sample_all.iloc[:,7].fillna(0)  #填充缺失值
#1.1~4.1期间  用户在该店铺使用的优惠券个数
feature_2=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160401]
feature_2=pysqldf("select c0,c1,count(c2) as f2 from feature_2 group by c0,c1")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_2,how='left',on=['c0','c1'])
train_sample_all.iloc[:,8]=train_sample_all.iloc[:,8].fillna(0)
#1.1~4.1期间  用户在该店铺优惠券使用率
feature_3=train_sample_all.iloc[:,8]/train_sample_all.iloc[:,7]
feature_3=feature_3.fillna(0)
feature_3=pd.DataFrame({'f3':feature_3})
train_sample_all=pd.concat([train_sample_all,feature_3],axis=1)
#1.1~4.1期间  用户在该店铺15天内使用优惠券个数
#计算时间差-天 (datetime.datetime(2016,2,11)-datetime.datetime(2016,2,01)).days
model_off_train_1_tmp=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160401]
diffday=[]
for i in range(len(model_off_train_1_tmp.iloc[:,6])):
    tmp=(datetime.datetime(int(model_off_train_1_tmp.iloc[i,6][:4]),int(model_off_train_1_tmp.iloc[i,6][4:6]),int(model_off_train_1_tmp.iloc[i,6][6:]))-datetime.datetime(int(model_off_train_1_tmp.iloc[i,5][:4]),int(model_off_train_1_tmp.iloc[i,5][4:6]),int(model_off_train_1_tmp.iloc[i,5][6:]))).days
    diffday.append(tmp)
feature_4=pd.DataFrame({'f4':diffday})
#重新建立索引
tmp=pd.DataFrame(model_off_train_1_tmp)
tmp.index=[i for i in range(tmp.shape[0])]
feature_4=pd.concat([tmp,feature_4],axis=1)
feature_4=feature_4[feature_4['f4'].astype(int)<=15]
feature_4=pysqldf("select c0,c1,count(c2) as f4 from feature_4 group by c0,c1")
train_sample_all=pd.merge(train_sample_all,feature_4,how='left',on=['c0','c1'])
train_sample_all.iloc[:,10]=train_sample_all.iloc[:,10].fillna(0)
#用户使用优惠券在15日之内占总使用优惠券的比例
feature_5=train_sample_all.iloc[:,10]/train_sample_all.iloc[:,8]
feature_5=feature_5.fillna(0)
feature_5=pd.DataFrame({'f5':feature_5})
train_sample_all=pd.concat([train_sample_all,feature_5],axis=1)
#用户在该店铺普通消费次数
feature_6=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160401]
feature_6=pysqldf("select c0,c1,count(c2) as f6 from feature_6 group by c0,c1")
train_sample_all=pd.merge(train_sample_all,feature_6,how='left',on=['c0','c1'])
train_sample_all.iloc[:,12]=train_sample_all.iloc[:,12].fillna(0)
#用户是否买过
feature_7=train_sample_all.iloc[:,8]+train_sample_all.iloc[:,12]
tmp=[]
for i in range(len(feature_7)):
  if feature_7[i]>0 :
     tmp.append(1)
  else :
     tmp.append(0)
feature_7=pd.DataFrame({'f7':tmp})
train_sample_all=pd.concat([train_sample_all,feature_7],axis=1)
#用户是否买过买过2次以上
feature_8=train_sample_all.iloc[:,8]+train_sample_all.iloc[:,12]
tmp=[]
for i in range(len(feature_8)):
  if feature_8[i]>1 :
     tmp.append(1)
  else :
     tmp.append(0)
feature_8=pd.DataFrame({'f8':tmp})
train_sample_all=pd.concat([train_sample_all,feature_8],axis=1)
#用户对该商店的距离
feature_9=train_sample_all.iloc[:,4]
tmp=[]
for i in range(len(feature_9)):
  if feature_9[i]=="null":
     tmp.append(5)
  else:
     tmp.append(feature_9[i])
feature_9=pd.DataFrame({'f9':tmp})
train_sample_all=pd.concat([train_sample_all,feature_9],axis=1)
#打折率
# import numpy as np
# a = np.array(tmp2)
# np.median(a.astype(float)) 中位数 mean 均值 argmax 最大
feature_10=train_sample_all.iloc[:,3]
tmp=[]
for i in range(len(feature_10)):
  if ':' in feature_10[i] :
     rate=feature_10[i].split(':')
     tmp.append(round((1-float(rate[1])/float(rate[0])),2))
  elif   feature_10[i]=='null'  :
     tmp.append(0.91)  #null填充为0.91
  else :
     tmp.append(feature_10[i])
feature_10=pd.DataFrame({'f10':tmp})
train_sample_all=pd.concat([train_sample_all,feature_10],axis=1)
print 'train_sample_allend 用户-商店 特征提取完毕'
################################################################################################
#统计用户消费情况--限定时间内
# 用户领取优惠券的数量
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160401],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160401]])
feature_1=pysqldf("select c0,count(c2) as u1 from feature_1 group by c0")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
#缺少值
# train_sample_all[pd.isnull(train_sample_all.iloc[:,7])].head()  #查看缺失值
# train_sample_all[pd.isnull(train_sample_all['u1'])].head()
train_sample_all['u1']=train_sample_all['u1'].fillna(0)  #填充缺失值
# 用户使用优惠券的数量
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160401]
feature_1=pysqldf("select c0,count(c2) as u2 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u2']=train_sample_all['u2'].fillna(0)
# 用户对优惠券的使用率
feature_1=train_sample_all['u2']/train_sample_all['u1']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u3':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
#用户普通消费
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160401]
feature_1=pysqldf("select c0,count(c2) as u4 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u4']=train_sample_all['u4'].fillna(0)
# 用户普通消费和使用优惠券的比例
feature_1=train_sample_all['u4']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
tmp=[]
for i in range(len(feature_1)):
  if feature_1[i]>100:
    tmp.append(100)
  else :
    tmp.append(feature_1[i])
feature_1=pd.DataFrame({'u5':tmp})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 用户使用优惠券在15日之内使用优惠券个数
model_off_train_1_tmp=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160401]
diffday=[]
for i in range(len(model_off_train_1_tmp.iloc[:,6])):
    tmp=(datetime.datetime(int(model_off_train_1_tmp.iloc[i,6][:4]),int(model_off_train_1_tmp.iloc[i,6][4:6]),int(model_off_train_1_tmp.iloc[i,6][6:]))-datetime.datetime(int(model_off_train_1_tmp.iloc[i,5][:4]),int(model_off_train_1_tmp.iloc[i,5][4:6]),int(model_off_train_1_tmp.iloc[i,5][6:]))).days
    diffday.append(tmp)
feature_1=pd.DataFrame({'u6':diffday})
#重新建立索引
tmp=pd.DataFrame(model_off_train_1_tmp)
tmp.index=[i for i in range(tmp.shape[0])]
feature_1=pd.concat([tmp,feature_1],axis=1)
feature_1=feature_1[feature_1['u6'].astype(int)<=15]
feature_1=pysqldf("select c0,count(c2) as u6 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u6']=train_sample_all['u6'].fillna(0)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# 用户使用优惠券在15日之内占总使用优惠券的比例
feature_1=train_sample_all['u6']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u7':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
print 'train_sample_allend 用户 特征提取完毕'
###############################################################################################
# 统计商店消费情况--限定时间内
# 商店普通消费个数
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160401]
feature_1=pysqldf("select c1,count(c2) as m1 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m1']=train_sample_all['m1'].fillna(0)
# 商店使用优惠券的次数
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160401]
feature_1=pysqldf("select c1,count(c2) as m2 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m2']=train_sample_all['m2'].fillna(0)
# 商店发出优惠券的个数
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160401],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160401]])
feature_1=pysqldf("select c1,count(c2) as m3 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m3']=train_sample_all['m3'].fillna(0)
# 商店优惠券使用率
feature_1=train_sample_all['m2']/train_sample_all['m3']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m4':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 使用优惠券占总消费的比例
feature_1=train_sample_all['m2']/(train_sample_all['m1']+train_sample_all['m2'])
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m5':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
print 'train_sample_allend 商店 特征提取完毕'
###############################################################################################
#标签label  4.15-5.15购买过的
lable=train_sample_all.iloc[:,6]
tmp=[]
for i in range(len(lable)):
  if lable[i]=="null":
     tmp.append(0)
  elif int(lable[i])>=20160415:
     tmp.append(1)
  else:
     tmp.append(0)
lable=pd.DataFrame({'lab':tmp})
train_sample_all=pd.concat([train_sample_all,lable],axis=1)
train_sample_allend=train_sample_all
# train_sample_all.to_csv("../result/train_sample_all.csv",header=False,index=False)
print 'train_sample_allend 全部特征+标签 提取完毕'










###############################################################################################
#从领取时间计算  正样本1-负样本2-普通消费3  2.1~5.1--5.15-6.15
#正样本1  领取优惠券并且使用
model_off_train_1=off_train[(off_train.iloc[:,6]!="null")&(off_train.iloc[:,2]!="null")]
#负样本2  领取优惠券但未使用
model_off_train_2=off_train[(off_train.iloc[:,6]=="null")&(off_train.iloc[:,2]!="null")]
#普通消费3 未领取但购买  普通消费
model_off_train_3=off_train[(off_train.iloc[:,6]!="null")&(off_train.iloc[:,2]=="null")]

##model_off_train_1.dtypes 查看所有类型
train_sample_1=model_off_train_1[(model_off_train_1.iloc[:,6].astype(int)<=20160615)&(model_off_train_1.iloc[:,6].astype(int)>=20160201)]
train_sample_2=model_off_train_2[(model_off_train_2.iloc[:,5].astype(int)<=20160515)&(model_off_train_1.iloc[:,6].astype(int)>=20160201)]
train_sample_3=model_off_train_3[(model_off_train_3.iloc[:,6].astype(int)<=20160615)&(model_off_train_1.iloc[:,6].astype(int)>=20160201)]
train_sample_all=pd.concat([train_sample_1,train_sample_2,train_sample_3])#train_sample_all.shape
train_sample_all.columns=['c0','c1','c2','c3','c4','c5','c6']
model_off_train_1=train_sample_1
model_off_train_2=train_sample_2
model_off_train_3=train_sample_3

#统计特征
#期间  用户在该店铺领取的的优惠券个数
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160501],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160501]])
# feature_1.columns=['c0','c1','c2','c3','c4','c5','c6']
# feature_1=feature_1.groupby(['c0','c1'])['c2'].count() #sql代替group by
# pd.DataFrame({'c0':tmp2.index,'c1':tmp2.iloc[:,1]})
pysqldf=lambda q:sqldf(q,globals())
feature_1=pysqldf("select c0,c1,count(c2) as f1 from feature_1 group by c0,c1")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0','c1'])
#缺少值
# train_sample_all[pd.isnull(train_sample_all.iloc[:,7])].head()  #查看缺失值
train_sample_all.iloc[:,7]=train_sample_all.iloc[:,7].fillna(0)  #填充缺失值
#期间  用户在该店铺使用的优惠券个数
feature_2=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160501]
feature_2=pysqldf("select c0,c1,count(c2) as f2 from feature_2 group by c0,c1")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_2,how='left',on=['c0','c1'])
train_sample_all.iloc[:,8]=train_sample_all.iloc[:,8].fillna(0)
#期间  用户在该店铺优惠券使用率
feature_3=train_sample_all.iloc[:,8]/train_sample_all.iloc[:,7]
feature_3=feature_3.fillna(0)
feature_3=pd.DataFrame({'f3':feature_3})
train_sample_all=pd.concat([train_sample_all,feature_3],axis=1)
#期间  用户在该店铺15天内使用优惠券个数
#计算时间差-天 (datetime.datetime(2016,2,11)-datetime.datetime(2016,2,01)).days
model_off_train_1_tmp=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160401]
diffday=[]
for i in range(len(model_off_train_1_tmp.iloc[:,6])):
    tmp=(datetime.datetime(int(model_off_train_1_tmp.iloc[i,6][:4]),int(model_off_train_1_tmp.iloc[i,6][4:6]),int(model_off_train_1_tmp.iloc[i,6][6:]))-datetime.datetime(int(model_off_train_1_tmp.iloc[i,5][:4]),int(model_off_train_1_tmp.iloc[i,5][4:6]),int(model_off_train_1_tmp.iloc[i,5][6:]))).days
    diffday.append(tmp)
feature_4=pd.DataFrame({'f4':diffday})
#重新建立索引
tmp=pd.DataFrame(model_off_train_1_tmp)
tmp.index=[i for i in range(tmp.shape[0])]
feature_4=pd.concat([tmp,feature_4],axis=1)
feature_4=feature_4[feature_4['f4'].astype(int)<=15]
feature_4=pysqldf("select c0,c1,count(c2) as f4 from feature_4 group by c0,c1")
train_sample_all=pd.merge(train_sample_all,feature_4,how='left',on=['c0','c1'])
train_sample_all.iloc[:,10]=train_sample_all.iloc[:,10].fillna(0)
#用户使用优惠券在15日之内占总使用优惠券的比例
feature_5=train_sample_all.iloc[:,10]/train_sample_all.iloc[:,8]
feature_5=feature_5.fillna(0)
feature_5=pd.DataFrame({'f5':feature_5})
train_sample_all=pd.concat([train_sample_all,feature_5],axis=1)
#用户在该店铺普通消费次数
feature_6=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160501]
feature_6=pysqldf("select c0,c1,count(c2) as f6 from feature_6 group by c0,c1")
train_sample_all=pd.merge(train_sample_all,feature_6,how='left',on=['c0','c1'])
train_sample_all.iloc[:,12]=train_sample_all.iloc[:,12].fillna(0)
#用户是否买过
feature_7=train_sample_all.iloc[:,8]+train_sample_all.iloc[:,12]
tmp=[]
for i in range(len(feature_7)):
  if feature_7[i]>0 :
     tmp.append(1)
  else :
     tmp.append(0)
feature_7=pd.DataFrame({'f7':tmp})
train_sample_all=pd.concat([train_sample_all,feature_7],axis=1)
#用户是否买过买过2次以上
feature_8=train_sample_all.iloc[:,8]+train_sample_all.iloc[:,12]
tmp=[]
for i in range(len(feature_8)):
  if feature_8[i]>1 :
     tmp.append(1)
  else :
     tmp.append(0)
feature_8=pd.DataFrame({'f8':tmp})
train_sample_all=pd.concat([train_sample_all,feature_8],axis=1)
#用户对该商店的距离
feature_9=train_sample_all.iloc[:,4]
tmp=[]
for i in range(len(feature_9)):
  if feature_9[i]=="null":
     tmp.append(5)
  else:
     tmp.append(feature_9[i])
feature_9=pd.DataFrame({'f9':tmp})
train_sample_all=pd.concat([train_sample_all,feature_9],axis=1)
#打折率
feature_10=train_sample_all.iloc[:,3]
tmp=[]
for i in range(len(feature_10)):
  if ':' in feature_10[i] :
     rate=feature_10[i].split(':')
     tmp.append(round((1-float(rate[1])/float(rate[0])),2))
  elif   feature_10[i]=='null'  :
     tmp.append(0.91)  #null填充为0.91
  else :
     tmp.append(feature_10[i])
feature_10=pd.DataFrame({'f10':tmp})
train_sample_all=pd.concat([train_sample_all,feature_10],axis=1)
print 'test_sample_allend 用户-商店 特征提取完毕'
################################################################################################
#统计用户消费情况--限定时间内
# 用户领取优惠券的数量
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160501],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160501]])
feature_1=pysqldf("select c0,count(c2) as u1 from feature_1 group by c0")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
#缺少值
# train_sample_all[pd.isnull(train_sample_all.iloc[:,7])].head()  #查看缺失值
# train_sample_all[pd.isnull(train_sample_all['u1'])].head()
train_sample_all['u1']=train_sample_all['u1'].fillna(0)  #填充缺失值
# 用户使用优惠券的数量
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160501]
feature_1=pysqldf("select c0,count(c2) as u2 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u2']=train_sample_all['u2'].fillna(0)
# 用户对优惠券的使用率
feature_1=train_sample_all['u2']/train_sample_all['u1']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u3':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
#用户普通消费
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160501]
feature_1=pysqldf("select c0,count(c2) as u4 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u4']=train_sample_all['u4'].fillna(0)
# 用户普通消费和使用优惠券的比例
feature_1=train_sample_all['u4']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
tmp=[]
for i in range(len(feature_1)):
  if feature_1[i]>100:
    tmp.append(100)
  else :
    tmp.append(feature_1[i])
feature_1=pd.DataFrame({'u5':tmp})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 用户使用优惠券在15日之内使用优惠券个数
model_off_train_1_tmp=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160501]
diffday=[]
for i in range(len(model_off_train_1_tmp.iloc[:,6])):
    tmp=(datetime.datetime(int(model_off_train_1_tmp.iloc[i,6][:4]),int(model_off_train_1_tmp.iloc[i,6][4:6]),int(model_off_train_1_tmp.iloc[i,6][6:]))-datetime.datetime(int(model_off_train_1_tmp.iloc[i,5][:4]),int(model_off_train_1_tmp.iloc[i,5][4:6]),int(model_off_train_1_tmp.iloc[i,5][6:]))).days
    diffday.append(tmp)
feature_1=pd.DataFrame({'u6':diffday})
#重新建立索引
tmp=pd.DataFrame(model_off_train_1_tmp)
tmp.index=[i for i in range(tmp.shape[0])]
feature_1=pd.concat([tmp,feature_1],axis=1)
feature_1=feature_1[feature_1['u6'].astype(int)<=15]
feature_1=pysqldf("select c0,count(c2) as u6 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u6']=train_sample_all['u6'].fillna(0)
# 用户使用优惠券在15日之内占总使用优惠券的比例
feature_1=train_sample_all['u6']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u7':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
print 'test_sample_allend 用户 特征提取完毕'
###############################################################################################
# 统计商店消费情况--限定时间内
# 商店普通消费个数
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160501]
feature_1=pysqldf("select c1,count(c2) as m1 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m1']=train_sample_all['m1'].fillna(0)
# 商店使用优惠券的次数
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160501]
feature_1=pysqldf("select c1,count(c2) as m2 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m2']=train_sample_all['m2'].fillna(0)
# 商店发出优惠券的个数
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160501],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160501]])
feature_1=pysqldf("select c1,count(c2) as m3 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m3']=train_sample_all['m3'].fillna(0)
# 商店优惠券使用率
feature_1=train_sample_all['m2']/train_sample_all['m3']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m4':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 使用优惠券占总消费的比例
feature_1=train_sample_all['m2']/(train_sample_all['m1']+train_sample_all['m2'])
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m5':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
print 'test_sample_allend 商店 特征提取完毕'
###############################################################################################
#标签label  4.15-5.15购买过的
lable=train_sample_all.iloc[:,6]
tmp=[]
for i in range(len(lable)):
  if lable[i]=="null":
     tmp.append(0)
  elif int(lable[i])>=20160515:
     tmp.append(1)
  else:
     tmp.append(0)
lable=pd.DataFrame({'lab':tmp})
train_sample_all=pd.concat([train_sample_all,lable],axis=1)
# train_sample_all.to_csv("../result/test_sample_all.csv",header=False,index=False)
test_sample_allend=train_sample_all
print 'test_sample_allend 全部特征+标签 提取完毕'
del train_sample_all








###############################################################################################
##测试集--测试集分成 有用户商店信息41270  无用户商店信息72370--需要统计商家-用户各自的特征！！ 选择时间?滑动窗口？
uidmid=off_test.iloc[:,0:3]
uidmid=uidmid.drop_duplicates(['c0','c1'])
test_pre_all=pd.merge(off_train,uidmid,how='left',on=['c0','c1'])
test_pre_all.columns=['c0','c1','c2','c3','c4','c5','c6','c7']
test_pre_all_1=test_pre_all[-pd.isnull(test_pre_all.iloc[:,7])]
test_pre_all_2=test_pre_all[pd.isnull(test_pre_all.iloc[:,7])]
del test_pre_all
test_pre_all_1=test_pre_all_1.iloc[:,0:7]
#从领取时间计算  正样本1-负样本2-普通消费3  3.15~6.15--7.1-7.30
#正样本1  领取优惠券并且使用
model_off_train_1=test_pre_all_1[(test_pre_all_1.iloc[:,6]!="null")&(test_pre_all_1.iloc[:,2]!="null")]
#负样本2  领取优惠券但未使用
model_off_train_2=test_pre_all_1[(test_pre_all_1.iloc[:,6]=="null")&(test_pre_all_1.iloc[:,2]!="null")]
#普通消费3 未领取但购买  普通消费
model_off_train_3=test_pre_all_1[(test_pre_all_1.iloc[:,6]!="null")&(test_pre_all_1.iloc[:,2]=="null")]

##3.15~6.15--7.1-7.30 #model_off_train_1.dtypes 查看所有类型
train_sample_1=model_off_train_1[(model_off_train_1.iloc[:,6].astype(int)<=20160701)&(model_off_train_1.iloc[:,6].astype(int)>=20160315)]
train_sample_2=model_off_train_2[(model_off_train_2.iloc[:,5].astype(int)<=20160701)&(model_off_train_1.iloc[:,6].astype(int)>=20160315)]
train_sample_3=model_off_train_3[(model_off_train_3.iloc[:,6].astype(int)<=20160701)&(model_off_train_1.iloc[:,6].astype(int)>=20160315)]
train_sample_all=pd.concat([train_sample_1,train_sample_2,train_sample_3])#train_sample_all.shape
train_sample_all.columns=['c0','c1','c2','c3','c4','c5','c6']
model_off_train_1=train_sample_1
model_off_train_2=train_sample_2
model_off_train_3=train_sample_3
#统计特征
#期间  用户在该店铺领取的的优惠券个数
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160615],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160615]])
# feature_1.columns=['c0','c1','c2','c3','c4','c5','c6']
# feature_1=feature_1.groupby(['c0','c1'])['c2'].count() #sql代替group by
pysqldf=lambda q:sqldf(q,globals())
feature_1=pysqldf("select c0,c1,count(c2) as f1 from feature_1 group by c0,c1")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0','c1'])
#缺少值
# train_sample_all[pd.isnull(train_sample_all.iloc[:,7])].head()  #查看缺失值
train_sample_all.iloc[:,7]=train_sample_all.iloc[:,7].fillna(0)  #填充缺失值
#期间  用户在该店铺使用的优惠券个数
feature_2=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
feature_2=pysqldf("select c0,c1,count(c2) as f2 from feature_2 group by c0,c1")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_2,how='left',on=['c0','c1'])
train_sample_all.iloc[:,8]=train_sample_all.iloc[:,8].fillna(0)
#期间  用户在该店铺优惠券使用率
feature_3=train_sample_all.iloc[:,8]/train_sample_all.iloc[:,7]
feature_3=feature_3.fillna(0)
feature_3=pd.DataFrame({'f3':feature_3})
train_sample_all=pd.concat([train_sample_all,feature_3],axis=1)
#期间  用户在该店铺15天内使用优惠券个数
#计算时间差-天 (datetime.datetime(2016,2,11)-datetime.datetime(2016,2,01)).days
model_off_train_1_tmp=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
diffday=[]
for i in range(len(model_off_train_1_tmp.iloc[:,6])):
    tmp=(datetime.datetime(int(model_off_train_1_tmp.iloc[i,6][:4]),int(model_off_train_1_tmp.iloc[i,6][4:6]),int(model_off_train_1_tmp.iloc[i,6][6:]))-datetime.datetime(int(model_off_train_1_tmp.iloc[i,5][:4]),int(model_off_train_1_tmp.iloc[i,5][4:6]),int(model_off_train_1_tmp.iloc[i,5][6:]))).days
    diffday.append(tmp)
feature_4=pd.DataFrame({'f4':diffday})
#重新建立索引
tmp=pd.DataFrame(model_off_train_1_tmp)
tmp.index=[i for i in range(tmp.shape[0])]
feature_4=pd.concat([tmp,feature_4],axis=1)
feature_4=feature_4[feature_4['f4'].astype(int)<=15]
feature_4=pysqldf("select c0,c1,count(c2) as f4 from feature_4 group by c0,c1")
train_sample_all=pd.merge(train_sample_all,feature_4,how='left',on=['c0','c1'])
train_sample_all.iloc[:,10]=train_sample_all.iloc[:,10].fillna(0)
#用户使用优惠券在15日之内占总使用优惠券的比例
feature_5=train_sample_all.iloc[:,10]/train_sample_all.iloc[:,8]
feature_5=feature_5.fillna(0)
feature_5=pd.DataFrame({'f5':feature_5})
train_sample_all=pd.concat([train_sample_all,feature_5],axis=1)
#用户在该店铺普通消费次数
feature_6=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160615]
feature_6=pysqldf("select c0,c1,count(c2) as f6 from feature_6 group by c0,c1")
train_sample_all=pd.merge(train_sample_all,feature_6,how='left',on=['c0','c1'])
train_sample_all.iloc[:,12]=train_sample_all.iloc[:,12].fillna(0)
#用户是否买过
feature_7=train_sample_all.iloc[:,8]+train_sample_all.iloc[:,12]
tmp=[]
for i in range(len(feature_7)):
  if feature_7[i]>0 :
     tmp.append(1)
  else :
     tmp.append(0)
feature_7=pd.DataFrame({'f7':tmp})
train_sample_all=pd.concat([train_sample_all,feature_7],axis=1)
#用户是否买过买过2次以上
feature_8=train_sample_all.iloc[:,8]+train_sample_all.iloc[:,12]
tmp=[]
for i in range(len(feature_8)):
  if feature_8[i]>1 :
     tmp.append(1)
  else :
     tmp.append(0)
feature_8=pd.DataFrame({'f8':tmp})
train_sample_all=pd.concat([train_sample_all,feature_8],axis=1)
#用户对该商店的距离
feature_9=train_sample_all.iloc[:,4]
tmp=[]
for i in range(len(feature_9)):
  if feature_9[i]=="null":
     tmp.append(5)
  else:
     tmp.append(feature_9[i])
feature_9=pd.DataFrame({'f9':tmp})
train_sample_all=pd.concat([train_sample_all,feature_9],axis=1)
#打折率
feature_10=train_sample_all.iloc[:,3]
tmp=[]
for i in range(len(feature_10)):
  if ':' in feature_10[i] :
     rate=feature_10[i].split(':')
     tmp.append(round((1-float(rate[1])/float(rate[0])),2))
  elif   feature_10[i]=='null'  :
     tmp.append(0.91)  #null填充为0.91
  else :
     tmp.append(feature_10[i])
feature_10=pd.DataFrame({'f10':tmp})
train_sample_all=pd.concat([train_sample_all,feature_10],axis=1)
print 'test_pre_allend  用户-商店 特征提取完毕'
################################################################################################
#统计用户消费情况--限定时间内
# 用户领取优惠券的数量
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160615],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160615]])
feature_1=pysqldf("select c0,count(c2) as u1 from feature_1 group by c0")
#左连接
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
#缺少值
# train_sample_all[pd.isnull(train_sample_all.iloc[:,7])].head()  #查看缺失值
# train_sample_all[pd.isnull(train_sample_all['u1'])].head()
train_sample_all['u1']=train_sample_all['u1'].fillna(0)  #填充缺失值
# 用户使用优惠券的数量
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c0,count(c2) as u2 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u2']=train_sample_all['u2'].fillna(0)
# 用户对优惠券的使用率
feature_1=train_sample_all['u2']/train_sample_all['u1']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u3':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
#用户普通消费
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c0,count(c2) as u4 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u4']=train_sample_all['u4'].fillna(0)
# 用户普通消费和使用优惠券的比例
feature_1=train_sample_all['u4']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
tmp=[]
for i in range(len(feature_1)):
  if feature_1[i]>100:
    tmp.append(100)
  else :
    tmp.append(feature_1[i])
feature_1=pd.DataFrame({'u5':tmp})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 用户使用优惠券在15日之内使用优惠券个数
model_off_train_1_tmp=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
diffday=[]
for i in range(len(model_off_train_1_tmp.iloc[:,6])):
    tmp=(datetime.datetime(int(model_off_train_1_tmp.iloc[i,6][:4]),int(model_off_train_1_tmp.iloc[i,6][4:6]),int(model_off_train_1_tmp.iloc[i,6][6:]))-datetime.datetime(int(model_off_train_1_tmp.iloc[i,5][:4]),int(model_off_train_1_tmp.iloc[i,5][4:6]),int(model_off_train_1_tmp.iloc[i,5][6:]))).days
    diffday.append(tmp)
feature_1=pd.DataFrame({'u6':diffday})
#重新建立索引
tmp=pd.DataFrame(model_off_train_1_tmp)
tmp.index=[i for i in range(tmp.shape[0])]
feature_1=pd.concat([tmp,feature_1],axis=1)
feature_1=feature_1[feature_1['u6'].astype(int)<=15]
feature_1=pysqldf("select c0,count(c2) as u6 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u6']=train_sample_all['u6'].fillna(0)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# 用户使用优惠券在15日之内占总使用优惠券的比例
feature_1=train_sample_all['u6']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u7':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
print 'test_pre_allend 用户 特征提取完毕'
###############################################################################################
# 统计商店消费情况--限定时间内3.15~6.15--7.1-7.30
# 商店普通消费个数
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c1,count(c2) as m1 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m1']=train_sample_all['m1'].fillna(0)
# 商店使用优惠券的次数
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c1,count(c2) as m2 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m2']=train_sample_all['m2'].fillna(0)
# 商店发出优惠券的个数
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160615],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160615]])
feature_1=pysqldf("select c1,count(c2) as m3 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m3']=train_sample_all['m3'].fillna(0)
# 商店优惠券使用率
feature_1=train_sample_all['m2']/train_sample_all['m3']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m4':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 使用优惠券占总消费的比例
feature_1=train_sample_all['m2']/(train_sample_all['m1']+train_sample_all['m2'])
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m5':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
test_pre_allend=train_sample_all
print 'test_pre_allend 商店 特征提取完毕'
print 'test_pre_allend 所有特征 提取完毕'
###############################################################################################
uid=test_pre_all_2.drop_duplicates(['c0'])
uid=uid.iloc[:,0:1]
mid=test_pre_all_2.drop_duplicates(['c1'])
mid=mid.iloc[:,1:2]
#统计用户消费情况--限定时间内
# 用户领取优惠券的数量
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160615],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160615]])
feature_1=pysqldf("select c0,count(c2) as u1 from feature_1 group by c0")
#左连接
train_sample_all=pd.merge(uid,feature_1,how='left',on=['c0'])
#缺少值
# train_sample_all[pd.isnull(train_sample_all.iloc[:,7])].head()  #查看缺失值
# train_sample_all[pd.isnull(train_sample_all['u1'])].head()
train_sample_all['u1']=train_sample_all['u1'].fillna(0)  #填充缺失值
# 用户使用优惠券的数量
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c0,count(c2) as u2 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u2']=train_sample_all['u2'].fillna(0)
# 用户对优惠券的使用率
feature_1=train_sample_all['u2']/train_sample_all['u1']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u3':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
#用户普通消费
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c0,count(c2) as u4 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u4']=train_sample_all['u4'].fillna(0)
# 用户普通消费和使用优惠券的比例
feature_1=train_sample_all['u4']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
tmp=[]
for i in range(len(feature_1)):
  if feature_1[i]>100:
    tmp.append(100)
  else :
    tmp.append(feature_1[i])
feature_1=pd.DataFrame({'u5':tmp})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 用户使用优惠券在15日之内使用优惠券个数
model_off_train_1_tmp=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
diffday=[]
for i in range(len(model_off_train_1_tmp.iloc[:,6])):
    tmp=(datetime.datetime(int(model_off_train_1_tmp.iloc[i,6][:4]),int(model_off_train_1_tmp.iloc[i,6][4:6]),int(model_off_train_1_tmp.iloc[i,6][6:]))-datetime.datetime(int(model_off_train_1_tmp.iloc[i,5][:4]),int(model_off_train_1_tmp.iloc[i,5][4:6]),int(model_off_train_1_tmp.iloc[i,5][6:]))).days
    diffday.append(tmp)
feature_1=pd.DataFrame({'u6':diffday})
#重新建立索引
tmp=pd.DataFrame(model_off_train_1_tmp)
tmp.index=[i for i in range(tmp.shape[0])]
feature_1=pd.concat([tmp,feature_1],axis=1)
feature_1=feature_1[feature_1['u6'].astype(int)<=15]
feature_1=pysqldf("select c0,count(c2) as u6 from feature_1 group by c0")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c0'])
train_sample_all['u6']=train_sample_all['u6'].fillna(0)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# 用户使用优惠券在15日之内占总使用优惠券的比例
feature_1=train_sample_all['u6']/train_sample_all['u2']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'u7':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
test_pre_allend_uid=train_sample_all
print 'test_pre_allend_uid 用户 特征提取完毕'
###############################################################################################
# 统计商店消费情况--限定时间内3.15~6.15--7.1-7.30
# 商店普通消费个数
feature_1=model_off_train_3[model_off_train_3.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c1,count(c2) as m1 from feature_1 group by c1")
train_sample_all=pd.merge(mid,feature_1,how='left',on=['c1'])
train_sample_all['m1']=train_sample_all['m1'].fillna(0)
# 商店使用优惠券的次数
feature_1=model_off_train_1[model_off_train_1.iloc[:,6].astype(int)<=20160615]
feature_1=pysqldf("select c1,count(c2) as m2 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m2']=train_sample_all['m2'].fillna(0)
# 商店发出优惠券的个数
feature_1=pd.concat([model_off_train_1[model_off_train_1.iloc[:,5].astype(int)<=20160615],model_off_train_2[model_off_train_2.iloc[:,5].astype(int)<=20160615]])
feature_1=pysqldf("select c1,count(c2) as m3 from feature_1 group by c1")
train_sample_all=pd.merge(train_sample_all,feature_1,how='left',on=['c1'])
train_sample_all['m3']=train_sample_all['m3'].fillna(0)
# 商店优惠券使用率
feature_1=train_sample_all['m2']/train_sample_all['m3']
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m4':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
# 使用优惠券占总消费的比例
feature_1=train_sample_all['m2']/(train_sample_all['m1']+train_sample_all['m2'])
feature_1=feature_1.fillna(0)
feature_1=pd.DataFrame({'m5':feature_1})
train_sample_all=pd.concat([train_sample_all,feature_1],axis=1)
test_pre_allend_mid=train_sample_all
print 'test_pre_allend_mid 商店 特征提取完毕'
print 'test_pre_allend uid mid所有特征 提取完毕'


###############################################################################################
#连接特征
tmp=test_pre_allend.iloc[:,[0,1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]]
tmp=tmp.drop_duplicates(['c0','c1','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','u1','u2','u3','u4','u5','u6','u7','m1','m2','m3','m4','m5'])
test_pre_allend=pd.merge(off_test,tmp,how='left',on=['c0','c1'])
#按照有无信息切割--用户商店连接  用户连接  商店连接  全无连接
test_pre_allend_1=test_pre_allend[pd.isnull(test_pre_allend.iloc[:,7])]
test_pre_allend=test_pre_allend[-pd.isnull(test_pre_allend.iloc[:,7])]
test_pre_allend_1=test_pre_allend_1.iloc[:,0:6]
#连接用户
test_pre_uid=pd.merge(off_test,test_pre_allend_uid,how='left',on=['c0'])
# test_pre_uid[pd.isnull(test_pre_uid['u1'])].head()
test_pre_uidmid=pd.merge(test_pre_uid,test_pre_allend_mid,how='left',on=['c1'])
#填充所有缺失值为0
test_pre_uidmid=test_pre_uidmid[pd.isnull(test_pre_uidmid['u1'])].fillna(0)
########添加无用户商户信息的特征
index=[]
tmpx=[]
for i in range(test_pre_uidmid.shape[0]):
  tmpx.append(0)
  index.append(i)
test_pre_uidmid.index=index
feature_1=pd.DataFrame({'f1':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
feature_1=pd.DataFrame({'f2':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
feature_1=pd.DataFrame({'f3':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
feature_1=pd.DataFrame({'f4':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
feature_1=pd.DataFrame({'f5':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
feature_1=pd.DataFrame({'f6':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
feature_1=pd.DataFrame({'f7':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
feature_1=pd.DataFrame({'f8':tmpx})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_1],axis=1)
#用户对该商店的距离
feature_9=test_pre_uidmid.iloc[:,4]
tmp=[]
for i in range(len(feature_9)):
  if feature_9[i]=="null":
     tmp.append(5)
  else:
     tmp.append(feature_9[i])
feature_9=pd.DataFrame({'f9':tmp})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_9],axis=1)
#打折率
feature_10=test_pre_uidmid.iloc[:,3]
tmp=[]
for i in range(len(feature_10)):
  if ':' in feature_10[i] :
     rate=feature_10[i].split(':')
     tmp.append(round((1-float(rate[1])/float(rate[0])),2))
  elif   feature_10[i]=='null'  :
     tmp.append(0.91)  #null填充为0.91
  else :
     tmp.append(feature_10[i])
feature_10=pd.DataFrame({'f10':tmp})
test_pre_uidmid=pd.concat([test_pre_uidmid,feature_10],axis=1)
test_pre_allend_1=test_pre_uidmid.iloc[:,[0,1,2,3,4,5,18,19,20,21,22,23,24,25,26,27,6,7,8,9,10,11,12,13,14,15,16,17]]
#合并
# index=[]
# for i in range(test_pre_allend.shape[0]):
#   index.append(i)
# test_pre_allend.index=index
test_pre_allend_all=pd.concat([test_pre_allend,test_pre_allend_1])
#train_sample_allend
#test_sample_allend
#test_pre_allend
#test_pre_allend_1
#test_pre_allend_all
# test_pre_allend.to_csv("../result/test_pre_allend.csv",header=False,index=False)
# test_pre_allend_1.to_csv("../result/test_pre_allend_1.csv",header=False,index=False)
train_sample_allend.to_csv("../result/train_sample_allend.csv",header=False,index=False)
test_sample_allend.to_csv("../result/test_sample_allend.csv",header=False,index=False)
test_pre_allend_all.to_csv("../result/test_pre_allend_all.csv",header=False,index=False)
