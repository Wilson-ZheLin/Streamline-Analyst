from sklearn import metrics
from sklearn.metrics import roc_curve

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