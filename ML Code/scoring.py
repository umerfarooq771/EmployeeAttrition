## Importing Libraries
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

## Function
import dataPreProcessing
import modelEvaluation

warnings.filterwarnings('ignore')

print('\n****** Employee Attrition Case Study for Crayon (Model Training) ******')

## Reading Data from excel file

print('\n1 - Reading dataset')
maindata=pd.read_csv(r'employee-attrition.csv')

X_resampled,y_resampled=dataPreProcessing.dataPreProcessing(maindata)

## Spliting in test and train
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

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



