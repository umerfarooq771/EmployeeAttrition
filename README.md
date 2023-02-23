# Employee Attrition Case Study

## Introduction

Employee attrition refers to the process of losing employees who leave the organization voluntarily or involuntarily. Employee attrition can be a significant problem for organizations as it can lead to loss of productivity, decreased morale, and increased costs associated with recruiting, hiring, and training new employees.

In this case study, we will use machine learning techniques to predict employee attrition. We will use data from a fictional organization to develop a predictive model that can identify employees who are at high risk of leaving the organization.

## Dataset

The dataset used in this case study is a fictional dataset that contains information about employees at a fictional organization. The dataset contains 1,470 rows and 35 columns. Each row represents a different employee, and each column represents a different attribute of the employee. The dataset includes information such as age, job role, education, performance rating, and more.

## Goal

The goal of this case study is to develop a predictive model that can identify employees who are at high risk of leaving the organization. The model will be evaluated based on its accuracy, precision, recall, and f1 score.

## Approach

To achieve our goal, we will follow the following approach:  
Data preprocessing - We will preprocess the data to handle missing values, encode categorical variables, and scale the data.   
Modeling - We will train several machine learning models, including logistic regression, gradient boosting, XGBoost, and random forest. We will also train an ensemble model using a voting classifier.  
Evaluation - We will evaluate each model's performance using metrics such as accuracy, precision, recall, and f1 score.  
Deployment - We will deploy the best performing model to make predictions on new data. 

## The repository contains the following files:

### Assets
[employee-attrition.csv](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/employee-attrition.csv) - the dataset used in this case study.  
[requirement.txt](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/requirement.txt) - the file contains all the required libraries for this analysis.  
### EDA 
[EDA.ipynb](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/EDA.ipynb) - Used to perform EDA on the provided dataset. 
### Python training and scoring files
[mainTraining.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/main.py) - python code used for training and saving the models.   
[scoring.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/scoring.py) - python code used for reading models from pkl files and running on new data. 
### Data Pre-Processing
[findingCorrelation.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/findingCorrelation.py) - finding correlation between the independent varaibles in pair as well as target.  
[featureScaling.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/featureScaling.py) - applying feature scaling using min max scaler.  
[dataBalancing.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/dataBalancing.py) - applying over sampling with SMOTE.  
[dataPreProcessing.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/dataPreProcessing.py) - a Python script that contains the data preprocessing functions.  
### Models
[gradientBoosting.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/gradientBoosting.py) - a Python script that contains the gradient boosting model implementation.  
[logisticRegression.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/logisticRegression.py) - a Python script that contains the logistic regression model implementation.  
[modelEvaluation.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/modelEvaluation.py) - a Python script that contains the evaluation metrics implementation.  
[randomForest.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/randomForest.py) - a Python script that contains the random forest model implementation.  
[Xgboost.py](https://github.com/umerfarooq771/EmployeeAttrition/blob/main/ML%20Code/Xgboost.py) - a Python script that contains the XGBoost model implementation.  
[Trained Models](https://github.com/umerfarooq771/EmployeeAttrition/tree/main/ML%20Code/TrainedModels) - a folder that contains the trained models.  

README.md - a readme file that provides an overview of the case study.  

## Modeling Decision
### 1 - Going with min max scaler:
Min-max feature scaling is a commonly used technique for data normalization because it is easy to implement and is effective in handling features with different scales.  
1 - Intuitive interpretation: Min-max feature scaling produces values between 0 and 1, which are easy to interpret. This makes it easier to explain the results of the model to stakeholders.  

2 - Maintains the range of the original data: Min-max feature scaling preserves the range of the original data, which can be important for some use cases. For example, if the salary of an employee is an important feature for predicting attrition, preserving the range of the salary data may be important.  

3 - Works well with linear models: Min-max feature scaling is particularly effective with linear models, such as logistic regression. These models rely on a linear combination of the input features, and min-max scaling can help ensure that each feature contributes equally to the prediction.  

### 2 - Going with undersampling approch

if a binary classification model was overfitting on the original imbalanced dataset and undersampling did not help to mitigate the overfitting, but oversampling resulted in a more generalized model, it may suggest that the original dataset was too imbalanced for the model to learn a good representation of the minority class.  

In such cases, undersampling might have led to a loss of important information and therefore did not help the model to generalize well. On the other hand, oversampling can help the model to learn a better representation of the minority class by generating synthetic examples of the minority class. This can lead to a more balanced distribution of the classes and better generalization performance.  

However, it is also important to note that oversampling techniques can sometimes lead to overfitting if not used properly. For example, if oversampling generates too many synthetic examples that are very similar to the original minority class examples, it may lead to overfitting and poor generalization performance. Therefore, it is important to use appropriate oversampling techniques, such as SMOTE or Borderline-SMOTE, and to tune the hyperparameters to achieve better generalization performance.  

In summary, the justification for using oversampling over undersampling in such cases might be that the original dataset was too imbalanced for the model to learn a good representation of the minority class, and oversampling helped to improve the generalization performance of the model by generating synthetic examples of the minority class. However, it is important to use appropriate oversampling techniques and to tune the hyperparameters to avoid overfitting.  

### 3 - Going with Recall as evaluation matrix
If the cost of false positives (i.e., predicting an employee will leave when they actually stay) is higher, then precision may be more important. This is because precision measures the proportion of true positives (i.e., correctly identifying employees who will leave) among all predicted positives, so a higher precision means fewer false positives.  

On the other hand, if the cost of false negatives (i.e., predicting an employee will stay when they actually leave) is higher, then recall may be more important. This is because recall measures the proportion of true positives among all actual positives, so a higher recall means fewer false negatives.

### 4 - Keeping Logistic Regression as the final model
When deciding which model to use, it's important to consider both the training and testing performance. A model with high training performance but low testing performance may indicate overfitting, meaning the model is too complex and has memorized the training data, resulting in poor generalization to new data.  

In this case, the Random Forest, XG Boost model, Gradient boosting as well as ensemble have high recall on the training set, which could suggest overfitting. The logistic regression model, on the other hand, has similar recall scores on both the training and testing sets, indicating that it is not overfitting and is able to generalize well to new data.  

Furthermore, recall is a particularly important metric in the context of employee attrition because it measures the ability of the model to identify employees who are likely to leave the company. In this case, the logistic regression model has a high recall score on the testing set, meaning that it is able to identify a large proportion of employees who are likely to leave.  

In conclusion, based on the comparable performance on both training and testing sets and the high recall score on testing, the logistic regression model may be the best choice for this employee attrition case study.  


## Conclusion

In this case study, we developed a machine learning model to predict employee attrition. We trained several models and evaluated their performance using different evaluation metrics. We also developed an ensemble model using a voting classifier. The best model choosen forward which is good with generalization is Logistic Regression with a recall of 85.65 % on testing and 83.41 % on training.
