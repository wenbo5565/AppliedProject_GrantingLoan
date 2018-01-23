"""
This file contains functions and codes for data science take home challenge
Granting Loan.

A related project report including background information and result
can be found on github page at 

https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/Project_Report__Loan_Granting.pdf

Jan-2018
"""

import pandas as pd
import numpy as np
# import Helper 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import xgboost as xgb
plt.rcParams.update({'font.size': 14}) # set fontsize for label
""" --------------- Define some Helper function --------------- """

def xgboost_training(train_x,train_y,valid_x=None,
                     valid_y=None,eval_fun=None,
                     num_round=500,max_depth=3,eta=0.1,subsample=1,colsample=1):
    """ interface function to facilite cross validation """
    dtrain = xgb.DMatrix(data=train_x,label=train_y)
    dvalid = xgb.DMatrix(data=valid_x,label=valid_y)
    param = {'max_depth':max_depth,'eta':eta,'silent':1,'objective':'binary:logistic',
             'eval_metric':['logloss'],'subsample':subsample,'colsample_bytree':colsample}
    watchlist = [(dvalid,'eval'),(dtrain,'train')]
    num_round = num_round
    result_dict = {}
    bst = xgb.train(param,dtrain,num_round,watchlist,feval=eval_fun,evals_result=result_dict)
    xgb.plot_importance(bst,max_num_features=20)
    valid_score = result_dict['eval']['LoanGrantingScore'][-1]
    train_score = result_dict['train']['LoanGrantingScore'][-1]
    valid_logloss = result_dict['eval']['logloss'][-1]
    train_logloss = result_dict['train']['logloss'][-1]
    return {'train_logloss':train_logloss,'valid_logloss':valid_logloss,
            'train_score':train_score,'valid_score':valid_score}



def bank_performance(loan_repaid_vector):
    """ input a loan_repaid vector. output bank's performance
        and average performance per granted loan
    """
    performance = np.sum(loan_repaid_vector)-np.sum(loan_repaid_vector==0)
    performance_per_loan = performance/len(loan_repaid_vector)
    print(performance, np.round(performance_per_loan,3))

def mean_evalerror(preds,dtrain):
    """ customized evaluation function """
    labels = dtrain.get_label()
    predict_labels = (preds>=0.5) # >0.5 means more likely repay than not
    loss = -1.0*np.sum(np.logical_and(labels==0,predict_labels==1))
    gain = 1.0*np.sum(np.logical_and(labels==1,predict_labels==1))
    return 'LoanGrantingScore',(loss+gain)/len(labels)

def model_performance(true_label,predict_label):
    gain_index = np.logical_and(predict_label==1,true_label==1)
    loss_index = np.logical_and(predict_label==1,true_label==0)
    return 'model score is',np.sum(gain_index)-np.sum(loss_index)

""" --------------- Loading Data --------------- """

grant_data = pd.read_csv('nontest_data.csv')
test_data = pd.read_csv('test_data.csv')



N = len(grant_data)

""" -------------- Data Preprocessing and Feature Engineering ---------- """
# extract month and day information
loan_month = grant_data['date'].apply(lambda x: x[5:7])
loan_day = grant_data['date'].apply(lambda x: x[8:])
grant_data['day'] = loan_day.astype(int)
grant_data['month'] = loan_month.astype(int)

# labelencode and one-hot encode feature 'loan_purpose'
le=preprocessing.LabelEncoder()
le.fit(grant_data['loan_purpose'])
purpose_labelencoded = le.transform(grant_data['loan_purpose'].values)
oe = preprocessing.OneHotEncoder(sparse=False)
purpose_onehotencoded = oe.fit_transform(purpose_labelencoded.reshape(-1,1))
purpose_encoded_df = pd.DataFrame(purpose_onehotencoded,columns=list(le.classes_))
grant_data_before_concat = grant_data
grant_data_before_concat.reset_index(inplace=True)
grant_data = pd.concat([grant_data_before_concat,purpose_encoded_df],axis=1)

# rename a column
grant_data.rename(columns={'avg_percentage_credit_card_limit_used_last_year':'credit_limit_used'},inplace=True)

""" create a new variable based on three columns: is_first_loan,
    currently_repaying_loans and fully_repaid_previous_loans 
"""

""" encoding the group by order of their repaying rate (increasing) """
group3_index = (grant_data['is_first_loan']==1).values
bool1 = ~group3_index
bool2 = (grant_data['fully_repaid_previous_loans']==0).values
bool3 = (grant_data['currently_repaying_other_loans']==0).values
group1_index = np.logical_and.reduce((bool1,bool2,~bool3))
group2_index = np.logical_and.reduce((bool1,~bool2,~bool3))
group4_index = np.logical_and.reduce((bool1,bool2,bool3))
group5_index = np.logical_and.reduce((bool1,~bool2,bool3))

""" assign group information as a new feature """
grant_data = grant_data.assign(previous_loan_status=np.zeros(len(grant_data)))
grant_data['previous_loan_status'][group3_index]=3
grant_data['previous_loan_status'][group1_index]=1
grant_data['previous_loan_status'][group2_index]=2
grant_data['previous_loan_status'][group4_index]=4
grant_data['previous_loan_status'][group5_index]=5

""" --------------- Hyper-parameter Tuning and Cross Validation --------------- """


""" prepare label and features """
grant_y = grant_data['loan_repaid']
grant_x = grant_data.drop(['index','loan_repaid','loan_id','date','loan_granted', 
                           'is_first_loan','fully_repaid_previous_loans','currently_repaying_other_loans',
                           'loan_purpose'],axis=1)

""" create stratified K-Fold validation """
seed = 2017
skf = StratifiedKFold(n_splits=5,random_state=seed)

# max_depth = [2,3,4,5]
max_depth = 12
eta = 0.01
num_round = 800
col_sample=0.5
sub_sample=0.8

train_logloss = np.zeros(0)
valid_logloss = np.zeros(0)
train_score = np.zeros(0)
valid_score = np.zeros(0)

for train_index,valid_index in skf.split(grant_x,grant_y):
    train_x = grant_x.iloc[train_index]
    valid_x = grant_x.iloc[valid_index]
    train_y = grant_y.iloc[train_index]
    valid_y = grant_y.iloc[valid_index]
    print('parameter is num_round {}, max depth {}, eta {}'.format(num_round,max_depth,eta))
    result=xgboost_training(train_x,train_y,valid_x,valid_y,num_round=num_round,
                     eval_fun=mean_evalerror,eta=eta,max_depth=max_depth,
                     subsample=sub_sample,colsample=col_sample)
    print(result)
    train_logloss=np.append(train_logloss,result['train_logloss'])
    valid_logloss=np.append(valid_logloss,result['valid_logloss'])
    train_score=np.append(train_score,result['train_score'])
    valid_score=np.append(valid_score,result['valid_score'])
    # valid_result=np.append(valid_result,valid_eval_result)
# bank_performance(valid_y)
print('training set logloss is {0:.4f}'.format(train_logloss.mean()))
print('validation set logloss is {0:.4f}'.format(valid_logloss.mean()))

print('model performance on training set is {0:.2f}'.format(train_score.mean()))
print('model performance on validation set is {0:.2f}'.format(valid_score.mean()))


""" --- Training Final Model, Making Prediction and Evaluation Result --- """

dtrain = xgb.DMatrix(data=grant_x,label=grant_y)
param = {'max_depth':max_depth,'eta':eta,'silent':1,'objective':'binary:logistic',
         'eval_metric':['logloss'],'subsample':sub_sample,'colsample_bytree':col_sample}
watchlist = [(dtrain,'train')]
num_round = num_round
result_dict = {}
bst = xgb.train(param,dtrain,num_round,watchlist,feval=mean_evalerror,evals_result=result_dict)
xgb.plot_importance(bst,max_num_features=20)

""" preprocessing testing set to be dumped into model for prediction """
## date
loan_month = test_data['date'].apply(lambda x: x[5:7])
loan_day = test_data['date'].apply(lambda x: x[8:])
test_data['day'] = loan_day.astype(int)
test_data['month'] = loan_month.astype(int)

## one-hot encode loan purpose
purpose_labelencoded = le.transform(test_data['loan_purpose'].values)
purpose_onehotencoded = oe.transform(purpose_labelencoded.reshape(-1,1))
purpose_encoded_df = pd.DataFrame(purpose_onehotencoded,columns=list(le.classes_))
test_data_before_concat = test_data
test_data_before_concat.reset_index(inplace=True)
test_data = pd.concat([test_data_before_concat,purpose_encoded_df],axis=1)

## create "previous loan status"
group3_index = (test_data['is_first_loan']==1).values
bool1 = ~group3_index
bool2 = (test_data['fully_repaid_previous_loans']==0).values
bool3 = (test_data['currently_repaying_other_loans']==0).values
group1_index = np.logical_and.reduce((bool1,bool2,~bool3))
group2_index = np.logical_and.reduce((bool1,~bool2,~bool3))
group4_index = np.logical_and.reduce((bool1,bool2,bool3))
group5_index = np.logical_and.reduce((bool1,~bool2,bool3))
test_data = test_data.assign(previous_loan_status=np.zeros(len(test_data)))
test_data['previous_loan_status'][group3_index]=3
test_data['previous_loan_status'][group1_index]=1
test_data['previous_loan_status'][group2_index]=2
test_data['previous_loan_status'][group4_index]=4
test_data['previous_loan_status'][group5_index]=5

## rename a column
test_data.rename(columns={'avg_percentage_credit_card_limit_used_last_year':'credit_limit_used'},inplace=True)
## split label and features
test_y = test_data['loan_repaid']
test_x = test_data.drop(['index','loan_repaid','loan_id','date','loan_granted', 
                           'is_first_loan','fully_repaid_previous_loans','currently_repaying_other_loans',
                           'loan_purpose'],axis=1)

""" make prediction """
dtest = xgb.DMatrix(test_x)
ypred_prob = bst.predict(dtest) # output probability
ypred_label = (ypred_prob>=0.5).astype(float)

print("bank's score and mean score on testing set is ")
bank_performance(test_data['loan_repaid'])

print("model's score on testing set is ")
model_performance(test_y,ypred_label)
print("model's mean score on testing set is ")
model_performance(test_y,ypred_label)[1]/len(ypred_label)

""" ----- analyse impact of important variables -------- """

"""  check impact of 'is employed' """

print("number of employed borrowers in the testing set")
np.sum(test_x['is_employed']==1)
print("their application granting rate is")
(ypred_label[test_x['is_employed']==1]).mean()



print("number of unemployed borrowers in the testing set")
np.sum(test_x['is_employed']==0)
print("their application granting rate is")
(ypred_label[test_x['is_employed']==0]).mean()


## check prediction accuracy conditional on 'is_employed=1'
employed_granted_ind = np.logical_and(test_x['is_employed']==1,ypred_label==1)
test_y[employed_granted_ind].mean()

## check prediction accuracy conditional on 'is_employed=0'
unemployed_granted_ind = np.logical_and(test_x['is_employed']==0,ypred_label==1)
test_y[unemployed_granted_ind].mean()


""" check impact of 'checking amount' """
## prediction accuracy condition on checking amount
bins = np.arange(0,14000,1000)
label = np.arange(len(bins)-1)+1
test_data['binned_checking_amount']=pd.cut(test_data['checking_amount'],bins,labels=label,include_lowest=True)
test_data['binned_checking_amount']=test_data['binned_checking_amount'].astype(float)

model_granted_data_test = test_data[ypred_label==1]
condition_pred_acc=model_granted_data_test.groupby('binned_checking_amount')['loan_repaid'].mean()
plt.scatter(label,condition_pred_acc)

## loan granting percentage based on checking amount
ypred_label[test_x['checking_amount']<5000].mean()
ypred_label[test_x['checking_amount']>=5000].mean()
