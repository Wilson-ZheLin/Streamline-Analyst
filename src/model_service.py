from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

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

def split_data(X, Y, test_size = 0.2, random_state = 42, perform_pca = False):
    """
    Split data into training and test sets.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    print('Training data count: ', len(X_train))
    print('Testing data count: ', len(X_test))

    if not perform_pca:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test