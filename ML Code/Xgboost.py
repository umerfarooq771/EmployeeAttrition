
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def Xgboost(X_train, X_test, y_train, y_test):
    # Define the XGBoost classifier
    xgb = XGBClassifier(verbosity = 0,silent=True)
    # Define the parameter grid for cross-validation
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.3],
        'n_estimators': [50, 100, 200]
    }

    # Apply GridSearchCV for cross-validation
    grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search