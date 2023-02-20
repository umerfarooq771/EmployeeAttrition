
from sklearn.preprocessing import MinMaxScaler


def featureScaling(data_encoded,continuous_features):
    scaler = MinMaxScaler()
    scaler.fit(data_encoded[continuous_features])
    data_encoded[continuous_features]= scaler.transform(data_encoded[continuous_features]) 
    return data_encoded
