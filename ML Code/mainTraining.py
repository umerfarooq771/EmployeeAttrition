### This script is the main code to read data, apply pre-processing steps and further train and saving the ML models.


## Importing Libraries
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

## Function

import gradientBoosting
import logisticRegression
import modelEvaluation
import Xgboost
import randomForest
import dataPreProcessing
import random

random.seed(10)

warnings.filterwarnings('ignore')

print('\n****** Employee Attrition Case Study for Crayon (Model Training) ******')

## Reading Data from excel file

print('\n1 - Reading dataset')
maindata=pd.read_csv(r'employee-attrition.csv')

X_resampled,y_resampled=dataPreProcessing.dataPreProcessing(maindata)

## Spliting in test and train
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print('\n\n**Modeling**')

# print('\n1 - Applying Gradient Boosting')
# ## Gradient Booasting
# gbModel = gradientBoosting.gradientBoosting(X_train, X_test, y_train, y_test)
# y_pred = gbModel.predict(X_test)
# joblib.dump(gbModel, 'TrainedModels/gbModel.pkl')
# # Print the accuracy score, confusion matrix, precision, recall, and f1 score
# modelEvaluation.modelEvaluation(gbModel,y_pred,X_train, X_test, y_train, y_test)

print('\n2 - Applying Logistic Regression')
lrModel = logisticRegression.logisticRegression(X_train, X_test, y_train, y_test)
# Predict on the testing data using the best model
y_pred = lrModel.predict(X_test)
joblib.dump(lrModel, 'TrainedModels/lrModel.pkl')
# Print the accuracy score, confusion matrix, precision, recall, and f1 score
modelEvaluation.modelEvaluation(lrModel,y_pred,X_train, X_test, y_train, y_test)

# print('\n3 - Applying XG Boost')
# xGModel = Xgboost.Xgboost(X_train, X_test, y_train, y_test)
# # Predict on the testing data using the best model
# y_pred = xGModel.predict(X_test)
# joblib.dump(xGModel, 'TrainedModels/xGModel.pkl')
# # Print the accuracy score, confusion matrix, precision, recall, and f1 score
# modelEvaluation.modelEvaluation(xGModel,y_pred,X_train, X_test, y_train, y_test)

# print('\n4 - Applying Random Forest')
# rFModel = randomForest.randomForest(X_train, X_test, y_train, y_test)
# # Predict on the testing data using the best model
# y_pred = rFModel.predict(X_test)
# joblib.dump(rFModel, 'TrainedModels/rFModel.pkl')
# # Print the accuracy score, confusion matrix, precision, recall, and f1 score
# modelEvaluation.modelEvaluation(rFModel,y_pred,X_train, X_test, y_train, y_test)

# print('\n5 - Applying Ensemble (RF,xG,LR,GB)')
# ensemble = VotingClassifier(estimators=[('xG', xGModel), ('rF', rFModel), ('lr', lrModel), ('gb', gbModel)], voting='soft')
# # Predict on the testing data using the best model
# ensemble.fit(X_train, y_train)
# y_pred = ensemble.predict(X_test)
# joblib.dump(ensemble, 'TrainedModels/EnsembleModel.pkl')
# # Print the accuracy score, confusion matrix, precision, recall, and f1 score
# modelEvaluation.modelEvaluation(ensemble,y_pred,X_train, X_test, y_train, y_test)