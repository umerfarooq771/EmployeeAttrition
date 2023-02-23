## Importing Libraries
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
import random

random.seed(42)

## Function
import dataPreProcessing
import modelEvaluation

warnings.filterwarnings('ignore')

print('\n****** Employee Attrition Case Study for Crayon (Model Scoring) ******')

## Reading Data from excel file

print('\n1 - Reading dataset')
maindata=pd.read_csv(r'employee-attrition.csv')

X_resampled,y_resampled=dataPreProcessing.dataPreProcessing(maindata)

## Spliting in test and train
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

xGModel = joblib.load('TrainedModels/xGModel.pkl')
rFModel = joblib.load('TrainedModels/rFModel.pkl')
lrModel = joblib.load('TrainedModels/lrModel.pkl')
gbModel = joblib.load('TrainedModels/gbModel.pkl')
emsembleModel = joblib.load('TrainedModels/EnsembleModel.pkl')

print('\n Logistic Regression')
y_pred = lrModel.predict(X_test)
modelEvaluation.modelEvaluation(lrModel,y_pred,X_train, X_test, y_train, y_test)

print('\n Random Forest')
y_pred = rFModel.predict(X_test)
modelEvaluation.modelEvaluation(rFModel,y_pred,X_train, X_test, y_train, y_test)

print('\n Gradient Boosting')
y_pred = gbModel.predict(X_test)
modelEvaluation.modelEvaluation(lrModel,y_pred,X_train, X_test, y_train, y_test)

print('\n XgBoost')
y_pred = xGModel.predict(X_test)
modelEvaluation.modelEvaluation(xGModel,y_pred,X_train, X_test, y_train, y_test)

print('\n Ensemble (RF,xG,LR,GB)')
y_pred = emsembleModel.predict(X_test)
modelEvaluation.modelEvaluation(emsembleModel,y_pred,X_train, X_test, y_train, y_test)

## Model Selection
# When deciding which model to use, it's important to consider both the training and testing performance. 
# A model with high training performance but low testing performance may indicate overfitting, meaning the model 
# is too complex and has memorized the training data, resulting in poor generalization to new data.

# In this case, the Random Forest, XG Boost model, Gradient boosting as well as ensemble have high recall on the training set, 
# which could suggest overfitting. The logistic regression model, on the other hand, has similar recall scores on both the training 
# and testing sets, indicating that it is not overfitting and is able to generalize well to new data.

# Furthermore, recall is a particularly important metric in the context of employee attrition because it measures the ability 
# of the model to identify employees who are likely to leave the company. In this case, the logistic regression model has a 
# high recall score on the testing set, meaning that it is able to identify a large proportion of employees who are likely to leave.

# In conclusion, based on the comparable performance on both training and testing sets and the high recall score on 
# testing, the logistic regression model may be the best choice for this employee attrition case study.



