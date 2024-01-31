from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition, metrics
from sklearn.metrics import roc_curve

def KNN_train(X_train, Y_train, k=3):
    knn = KNeighborsClassifier()
    knn.fit(X_train,Y_train)
    return knn

def RandomForest_train(X_train, Y_train, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train, Y_train)
    return rf

def model_score(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    return score

def fpr_and_tpr(model, X_test, Y_test):
    Y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    return fpr, tpr

def auc(model, X_test, Y_test):
    fpr, tpr = fpr_and_tpr(model, X_test, Y_test)
    auc = metrics.auc(fpr, tpr)
    return auc