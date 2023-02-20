

## Importing Libraries


import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
from sklearn.utils import resample, shuffle

from sklearn.decomposition import PCA


## Function
import findingCorrelation
import featureScaling
import dataBalancing


print('\n** Employee Attrition Case Study for Crayon **')

## Reading Data from excel file
print('\n\n1 - Reading dataset')
maindata=pd.read_csv(r'employee-attrition.csv')

print('\n2 - Droping not needed columns')
colsToDrop = ['EmployeeCount', 'StandardHours','EmployeeNumber']
maindata = maindata.drop(colsToDrop, axis=1)

continuous_features = maindata.select_dtypes(include=['float64', 'int64']).columns.tolist()

print('\n3 - Encoding bianry columns to 1 and 0')
maindata['Attrition'] = maindata['Attrition'].replace({'Yes': 1, 'No': 0})
maindata['Gender'] = maindata['Gender'].replace({'Female':1, 'Male':0})
maindata['Over18'] = maindata['Over18'].replace({'Y':1, 'N':0})
maindata['OverTime'] = maindata['OverTime'].replace({'Yes':1, 'No':0})

### Identifying and removing features with more than 60% correlation with eachother and greater than 50% with the target
print('\n4 - Applying correlation')
OGColumnLen=len(maindata.columns)
maindata = findingCorrelation.findingCorrelation(maindata)
print('INFO - Dropped '+str(OGColumnLen-len(maindata.columns))+' columns from OG dataset')

## Identifying continous features
continuous_features = maindata.select_dtypes(include=['float64', 'int64']).columns.tolist()

## encoding columns where required
data_encoded = pd.get_dummies(maindata)
print('\n5 - Encoding columnns of dataset where required')

## Feature Scaling only on continous features
data_encoded=featureScaling.featureScaling(data_encoded,continuous_features)
print('\n6 - Feature Scaling on continuous features using min max scaler')
print('INFO - Total columns '+str(len(data_encoded.columns))+' , Now all columns are numeric')

## Checking for data balancing
yesCount = len(data_encoded.loc[data_encoded['Attrition'] == 1])
noCount = len(data_encoded.loc[data_encoded['Attrition'] == 0])
print('\n7 - Checking for data balancing')
print('INFO - Total events of Employees leaving the company are '+str(yesCount)+', Whereas Employees not leaving are '+str(noCount))

## Checking if balancing is required
if (yesCount/noCount) > 1:
    print('INFO - oversampling required for events where employees have not left')
    X_resampled,y_resampled  = dataBalancing.dataBalancing(data_encoded)
elif (noCount/yesCount) > 1:
    print('INFO - oversampling required for events where employees have left')
    X_resampled,y_resampled  = dataBalancing.dataBalancing(data_encoded)
else:
    print('INFO - Data is balanced for modeling')

noCount = y_resampled.value_counts()[0]
yesCount = y_resampled.value_counts()[1]
print('INFO - Total events of Employees leaving the company are '+str(yesCount)+', Whereas Employees not leaving are '+str(noCount))

## Spliting in test and train
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)