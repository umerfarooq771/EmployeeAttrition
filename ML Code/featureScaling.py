from sklearn.preprocessing import MinMaxScaler

## this function does min max feature scaling,

# Why used min max feature scaling : 
# Min-max feature scaling is a commonly used technique for data normalization because it is easy to implement and 
# is effective in handling features with different scales, best for Intuitive interpretation, Maintains the range 
# of the original data, Works well with linear models

def featureScaling(data_encoded,continuous_features):

    ## Initilizing min max scaler
    scaler = MinMaxScaler()
    scaler.fit(data_encoded[continuous_features])
    data_encoded[continuous_features]= scaler.transform(data_encoded[continuous_features]) 

    ## Returning the scaled dataframe
    return data_encoded
