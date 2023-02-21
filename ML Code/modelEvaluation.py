from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def modelEvaluation(model,y_pred,X_train, X_test, y_train, y_test):  
    print("\tAccuracy score: ", accuracy_score(y_test, y_pred))
    print("\tConfusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("\tPrecision: ", precision_score(y_test, y_pred))
    print("\tRecall: ", recall_score(y_test, y_pred))
    print("\tF1 score: ", f1_score(y_test, y_pred))
    y_pred_train = model.predict(X_train)
    print("\tAccuracy score (Training Set): ", accuracy_score(y_train, y_pred_train))
    print("\tPrecision: (Training Set)", precision_score(y_train, y_pred_train))
    print("\tRecall (Training Set): ", recall_score(y_train, y_pred_train))
