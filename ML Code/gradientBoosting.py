
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def gradientBoosting(X_train, X_test, y_train, y_test):

    # Define the gradient boosting model
    model = GradientBoostingClassifier()

    # Define the hyperparameter grid
    param_grid = {
        'learning_rate': [0.1, 0.05, 0.01],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5]
    }

    # Define the grid search object
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_