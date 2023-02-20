import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

def dataBalancing(data_encoded):
    X = data_encoded.drop('Attrition', axis=1)
    y = data_encoded['Attrition']

    # Instantiate the Borderline-SMOTE algorithm with default parameters
    smote = BorderlineSMOTE()

    # Fit and apply the Borderline-SMOTE algorithm to the dataset
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled,y_resampled
