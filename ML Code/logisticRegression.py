
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def logisticRegression(X_train, X_test, y_train, y_test):
    # Define the logistic regression model
    log_reg = LogisticRegression(penalty='l1',solver='liblinear', multi_class='ovr')

    # Define the hyperparameter grid
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

    # Define the cross-validation object
    cv = GridSearchCV(log_reg, param_grid, cv=5)

    # Fit the cross-validation object to the training data
    cv.fit(X_train, y_train)

    return cv


