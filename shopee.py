# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:38:24 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import time
np.random.seed(0)

#import train and predict data
data = pd.read_csv("D:/Shopee/reactivation_data/training.csv")
test_data = pd.read_csv("D:/Shopee/reactivation_data/predict.csv")
#import other information
transactions = pd.read_csv("D:/Shopee/reactivation_data/transactions_MY.csv")
users = pd.read_csv("D:/Shopee/reactivation_data/user_profiles_MY.csv")
likes =pd.read_csv("D:/Shopee/reactivation_data/likes.csv")
voucher_date = pd.read_csv("D:/Shopee/reactivation_data/voucher_distribution_active_date.csv")
voucher = pd.read_csv("D:/Shopee/reactivation_data/voucher_mechanics.csv")
#import behavior data
log=[]
for i in range(31):
    log.append(pd.read_csv("D:/Shopee/reactivation_data/view_log_" +str(i) + ".csv"))
    
used_y =data["used?"]
repurchase_y = data[["repurchase_15?","repurchase_30?","repurchase_60?","repurchase_90?"]]
X = data[["userid","promotionid_received","voucher_code_received","voucher_received_time"]]
repurchase_reform_y = repurchase_y.sum(1)


###############################################################################
#we extract the transaction information for each sample.
#Since we should not use the transactions after the voucher receiving date for 
#the prediction, we aggregate the number of transactions and the average price 
#of the previous transactions. 

def extract_transaction_information(row):
    userid = row['userid']
    t = row["voucher_received_time"]
    tmp = transactions.loc[ transactions['userid']==userid]
    tmp = tmp.loc[tmp['order_time'] < t]
    tmp = tmp["total_price"]
    num = tmp.shape[0]
    avg = tmp.sum()/num if num>0 else 0
    return [num,avg]

trans_extract = X[["userid","voucher_received_time"]].apply(extract_transaction_information,axis=1)
trans_extract.columns = ['num_orders', 'avg_price']
trans_extract.to_csv("D:/Shopee/reactivation_data/trans_extract.csv",index=False)
test_trans_extract = test_data[["userid","voucher_received_time"]].apply(extract_transaction_information,axis=1)
test_trans_extract.columns = ['num_orders', 'avg_price']
test_trans_extract.to_csv("D:/Shopee/reactivation_data/test_trans_extract.csv",index=False)
###############################################################################
#Similarly, we aggregate the information of likes and unlikes for each user.
def extract_likes_information(row):
    userid = row['userid']
    t = row["voucher_received_time"]
    tmp = likes.loc[ likes['userid']==userid]
    tmp = tmp.loc[tmp['ctime'] < t]
    tmp = tmp["status"]
    num_likes = tmp.sum()
    num_unlikes = tmp.shape[0] - num_likes
    return [num_likes ,num_unlikes]

likes_extract = X[["userid","voucher_received_time"]].apply(extract_likes_information,axis=1)
likes_extract.columns = ['likes', 'unlikes']
likes_extract.to_csv("D:/Shopee/reactivation_data/likes_extract.csv",index=False)
test_likes_extract = test_data[["userid","voucher_received_time"]].apply(extract_likes_information,axis=1)
test_likes_extract.columns = ['likes', 'unlikes']
test_likes_extract.to_csv("D:/Shopee/reactivation_data/test_likes_extract.csv",index=False)
###############################################################################

## Combine features

X = pd.merge(X, users,  how='left', left_on=['userid'],right_on=['userid']) #combine user info
X = pd.merge(X, voucher,  how='left', left_on=['promotionid_received'],right_on=['promotionid_received'])#combine voucher info
X = pd.merge(X, voucher_date,  how='left', left_on=['userid','promotionid_received','voucher_code_received','voucher_received_time'], right_on = ['userid','promotionid_received','voucher_code_received','voucher_received_time'])
#combine active session from voucher_distribution_active_date
del X['voucher_received_date']
X[trans_extract.columns]=trans_extract
X[likes_extract.columns]=likes_extract

def epoch2datetime(t,offset):
    lt = time.localtime(t-86400*offset)
    tt = time.strftime('%Y-%m-%d', lt)
    return tt
# combine the log info
for i in range(31):
    if i==0:
        date_colname = 'voucher_received_date'
    else:
        date_colname = str(i)+'_day_before'
    X["time"] =X["voucher_received_time"].apply(epoch2datetime,args=(i,)) 
        
    tmp = log[i].loc[log[i]["event_name"] == "trackGenericScroll" ,['userid', date_colname,'count']]
    tmp.rename(columns={'count': 'trackGenericScroll_'+str(i)}, inplace=True)
    X = pd.merge(X, tmp,  how='left', left_on=['userid','time'],right_on=['userid',date_colname])
    del X[date_colname]
    tmp = log[i].loc[log[i]["event_name"] == "trackGenericSearchPageView" ,['userid', date_colname,'count']]
    tmp.rename(columns={'count': 'trackGenericSearchPageView_'+str(i)}, inplace=True)
    X = pd.merge(X, tmp,  how='left', left_on=['userid','time'],right_on=['userid',date_colname])
    del X[date_colname]
    tmp = log[i].loc[log[i]["event_name"] == "trackGenericClick" ,['userid', date_colname,'count']]
    tmp.rename(columns={'count': 'trackGenericClick_'+str(i)}, inplace=True)
    X = pd.merge(X, tmp,  how='left', left_on=['userid','time'],right_on=['userid',date_colname])
    del X[date_colname]
del X["time"]
#save the result
X.to_csv("D:/Shopee/reactivation_data/features.csv",index=False)
X=pd.read_csv("D:/Shopee/reactivation_data/features.csv")
###############################################################################
# Deal with Missing data
# first deal with gender 
# 0 is male, 1 is predict male, 2 is missing, 3 is predict female, 4 is female
X.loc[X['gender']==1,'gender'] = 0
X.loc[X['gender']==3,'gender'] = 1
X.loc[X['gender']==4,'gender'] = 3
X.loc[X['gender']==2,'gender'] = 4
X['gender']=X['gender'].fillna(2)

# fill the missing data for the behavior data
X=X.fillna(0)
# scale the data
sd=[np.std(X['avg_price']),np.std(X['likes']),np.std(X['unlikes'])]
X['avg_price']=X['avg_price']/sd[0]
X['likes']=X['likes']/sd[1]
X['unlikes']=X['unlikes']/sd[2]
# transform the data, remove the skewness, using log(x+1)
for i in range(31):
    colname='active_'+str(i)
    X[colname]=np.log(X[colname]+1)
    colname='trackGenericScroll_'+str(i)
    X[colname]=np.log(X[colname]+1)
    colname='trackGenericSearchPageView_'+str(i)
    X[colname]=np.log(X[colname]+1)
    colname='trackGenericClick_'+str(i)
    X[colname]=np.log(X[colname]+1)
# calculate the age of user when receive voucher  
def age(row):
    vd = time.strftime('%Y', time.localtime(row['voucher_received_time']))
    bd = str(row['birthday'])[0:4]             
    return int(vd)-int(bd)    
X.loc[X['birthday']!=0,'age']=X.loc[X['birthday']!=0].apply(age,axis=1)
avg_age=int(np.mean(X.loc[X['birthday']!=0,'age']))
X.loc[X['birthday']==0,'age']=avg_age #imputer the missing data with average age
# calculate the acount_age of user when receive voucher  
def account_age(row):
    vd = time.strftime('%Y', time.localtime(row['voucher_received_time']))
    bd = str(row['registration_time'])[0:4]             
    return int(vd)-int(bd)
X.loc[:,'account_age']=X.apply(account_age,axis=1)
   
X.to_csv("D:/Shopee/reactivation_data/clean_features.csv",index=False)    
###############################################################################
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def plot_precision_recall_curve(test,score):    # precision_recall_curve
    average_precision = average_precision_score(test, score)
    precision, recall, _ = precision_recall_curve(test, score)    
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
def plot_confusion(test,pred): # plot confusion matrix
    plt.imshow(metrics.confusion_matrix(pred, test),
           interpolation='nearest', cmap=plt.cm.binary)
    plt.grid(False)
    plt.colorbar()
    plt.ylabel("predicted label")
    plt.xlabel("true label")

features=X.columns
features=np.delete(features,[0,1,2,5,9]) #remove birthday and registration_time

Xtrain, Xtest, used_ytrain, used_ytest,repurchase_reform_ytrain,repurchase_reform_ytest = train_test_split(X, used_y, repurchase_reform_y,test_size=0.1)
#Xtrain, Xtrain_lr, used_ytrain, used_ytrain_lr,repurchase_reform_ytrain,repurchase_reform_ytrain_lr = train_test_split(Xtrain, used_ytrain, repurchase_reform_ytrain,test_size=0.5)
used_rf = RandomForestClassifier(n_estimators=100, random_state=30)
used_rf.fit(Xtrain[features], used_ytrain)
used_yscore = used_rf.predict_proba(Xtest[features])[:, 1]
used_ypred = used_rf.predict(Xtest[features])
#threshold=0.3
#used_ypred = (used_yscore>threshold).astype(int)                            
metrics.accuracy_score(used_ypred, used_ytest)
metrics.confusion_matrix(used_ypred, used_ytest)
plot_precision_recall_curve(used_ytest,used_yscore)
plot_confusion(used_ytest,used_ypred)


rep_rf = RandomForestClassifier(n_estimators=100, random_state=30)
rep_rf.fit(Xtrain[features], repurchase_reform_ytrain)
repurchase_reform_ypred = rep_rf.predict(Xtest[features])
metrics.confusion_matrix(repurchase_reform_ypred, repurchase_reform_ytest)
plot_confusion(repurchase_reform_ytest,repurchase_reform_ypred)








###############################################################################
###############################################################################
###############################################################################
# Predict on test data
## Combine test data features
test_data = pd.merge(test_data, users,  how='left', left_on=['userid'],right_on=['userid'])
test_data = pd.merge(test_data, voucher,  how='left', left_on=['promotionid_received'],right_on=['promotionid_received'])
test_data = pd.merge(test_data, voucher_date,  how='left', left_on=['userid','promotionid_received','voucher_code_received','voucher_received_time'], right_on = ['userid','promotionid_received','voucher_code_received','voucher_received_time'])
del test_data['voucher_received_date']
test_data[test_trans_extract.columns]=test_trans_extract
test_data[test_likes_extract.columns]=test_likes_extract

for i in range(31):
    if i==0:
        date_colname = 'voucher_received_date'
    else:
        date_colname = str(i)+'_day_before'
    test_data["time"] =test_data["voucher_received_time"].apply(epoch2datetime,args=(i,)) 
        
    tmp = log[i].loc[log[i]["event_name"] == "trackGenericScroll" ,['userid', date_colname,'count']]
    tmp.rename(columns={'count': 'trackGenericScroll_'+str(i)}, inplace=True)
    test_data = pd.merge(test_data, tmp,  how='left', left_on=['userid','time'],right_on=['userid',date_colname])
    del test_data[date_colname]
    tmp = log[i].loc[log[i]["event_name"] == "trackGenericSearchPageView" ,['userid', date_colname,'count']]
    tmp.rename(columns={'count': 'trackGenericSearchPageView_'+str(i)}, inplace=True)
    test_data = pd.merge(test_data, tmp,  how='left', left_on=['userid','time'],right_on=['userid',date_colname])
    del test_data[date_colname]
    tmp = log[i].loc[log[i]["event_name"] == "trackGenericClick" ,['userid', date_colname,'count']]
    tmp.rename(columns={'count': 'trackGenericClick_'+str(i)}, inplace=True)
    test_data = pd.merge(test_data, tmp,  how='left', left_on=['userid','time'],right_on=['userid',date_colname])
    del test_data[date_colname]
del test_data["time"]
test_data.to_csv("D:/Shopee/reactivation_data/test_features.csv",index=False)
test_data=pd.read_csv("D:/Shopee/reactivation_data/test_features.csv")

## Data cleaning and transformation
test_data.loc[test_data['gender']==1,'gender'] = 0
test_data.loc[test_data['gender']==3,'gender'] = 1
test_data.loc[test_data['gender']==4,'gender'] = 3
test_data.loc[test_data['gender']==2,'gender'] = 4
test_data['gender']=test_data['gender'].fillna(2)

test_data=test_data.fillna(0)
test_data['avg_price']=test_data['avg_price']/sd[0]
test_data['likes']=test_data['likes']/sd[1]
test_data['unlikes']=test_data['unlikes']/sd[2]
for i in range(31):
    colname='active_'+str(i)
    test_data[colname]=np.log(test_data[colname]+1)
    colname='trackGenericScroll_'+str(i)
    test_data[colname]=np.log(test_data[colname]+1)
    colname='trackGenericSearchPageView_'+str(i)
    test_data[colname]=np.log(test_data[colname]+1)
    colname='trackGenericClick_'+str(i)
    test_data[colname]=np.log(test_data[colname]+1)

test_data.loc[test_data['birthday']!=0,'age']=test_data.loc[test_data['birthday']!=0].apply(age,axis=1)
test_data.loc[test_data['birthday']==0,'age']=avg_age #imputer the missing data with average age
test_data.loc[:,'account_age']=test_data.apply(account_age,axis=1)




used_test_pred = used_rf.predict(test_data[features])
repurchase_reform_test_pred = rep_rf.predict(test_data[features])
test = pd.read_csv("D:/Shopee/reactivation_data/predict.csv")
test["used?"]=used_test_pred
test["repurchase_15?"] = (repurchase_reform_test_pred>=4).astype(int)
test["repurchase_30?"] = (repurchase_reform_test_pred>=3).astype(int)
test["repurchase_60?"] = (repurchase_reform_test_pred>=2).astype(int)
test["repurchase_90?"] = (repurchase_reform_test_pred>=1).astype(int)
test.to_csv("D:/Shopee/reactivation_data/test.csv",index=False)
