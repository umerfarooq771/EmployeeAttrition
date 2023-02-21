import findingCorrelation
import featureScaling
import dataBalancing
import pandas as pd

def dataPreProcessing(maindata):
    print('\n\n**Data Preprocessing**')
    print('\n1 - Droping not needed columns')
    colsToDrop = ['EmployeeCount', 'StandardHours','EmployeeNumber']
    maindata = maindata.drop(colsToDrop, axis=1)

    continuous_features = maindata.select_dtypes(include=['float64', 'int64']).columns.tolist()

    print('\n2 - Encoding bianry columns to 1 and 0')
    maindata['Attrition'] = maindata['Attrition'].replace({'Yes': 1, 'No': 0})
    maindata['Gender'] = maindata['Gender'].replace({'Female':1, 'Male':0})
    maindata['Over18'] = maindata['Over18'].replace({'Y':1, 'N':0})
    maindata['OverTime'] = maindata['OverTime'].replace({'Yes':1, 'No':0})

    ### Identifying and removing features with more than 60% correlation with eachother and greater than 50% with the target
    print('\n3 - Applying correlation')
    OGColumnLen=len(maindata.columns)
    maindata = findingCorrelation.findingCorrelation(maindata)
    print('INFO - Dropped '+str(OGColumnLen-len(maindata.columns))+' columns from OG dataset')

    ## Identifying continous features
    continuous_features = maindata.select_dtypes(include=['float64', 'int64']).columns.tolist()

    ## encoding columns where required
    data_encoded = pd.get_dummies(maindata)
    print('\n4 - Encoding columnns of dataset where required')

    ## Feature Scaling only on continous features
    data_encoded=featureScaling.featureScaling(data_encoded,continuous_features)
    print('\n5 - Feature Scaling on continuous features using min max scaler')
    print('INFO - Total columns '+str(len(data_encoded.columns))+' , Now all columns are numeric')

    ## Checking for data balancing
    yesCount = len(data_encoded.loc[data_encoded['Attrition'] == 1])
    noCount = len(data_encoded.loc[data_encoded['Attrition'] == 0])
    print('\n6 - Checking for data balancing')
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
    return X_resampled,y_resampled