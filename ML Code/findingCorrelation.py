

def findingCorrelation(maindata):

    # Calculate the correlation matrix
    corr_matrix = maindata.corr()

    # Identify the features that have a high correlation with the target variable
    target_corr = corr_matrix['Attrition'].abs().sort_values(ascending=False)
    highly_corr_features = target_corr[target_corr > 0.5].index.tolist()
    highly_corr_features.remove('Attrition')

    # Identify the pairs of features that have a high correlation with each other
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    highly_corr_pairs = corr_pairs[(corr_pairs > 0.6) & (corr_pairs < 1.0)].index.to_list()
    highly_corr_pairs = [item for sublist in [list(t) for t in highly_corr_pairs] for item in (sublist if isinstance(sublist, list) else [sublist])]

    # Remove the highly correlated features from the DataFrame
    maindata = maindata.drop(columns=highly_corr_features+highly_corr_pairs)

    return maindata