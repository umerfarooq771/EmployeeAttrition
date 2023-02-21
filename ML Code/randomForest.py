
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def randomForest(X_train, X_test, y_train, y_test):

    # Define the hyperparameters to search over
    param_grid = {'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]}


    # Create a random forest classifier instance
    rfc = RandomForestClassifier(random_state=42)

    # Create a stratified 5-fold cross-validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create a Grid Search object to find the best hyperparameters
    grid_search = GridSearchCV(rfc, param_grid, cv=cv, n_jobs=-1)

    # Fit the Grid Search object to the training data
    grid_search.fit(X_train, y_train)

    return grid_search