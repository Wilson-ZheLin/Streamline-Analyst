from sklearn.linear_model import LinearRegression, Ridge, Lasso

def LinearRegression_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    lr = LinearRegression(**model_params)
    lr.fit(X_train, Y_train)
    return lr

def RidgeRegression_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    ridge = Ridge(**model_params)
    ridge.fit(X_train, Y_train)
    return ridge

def LassoRegression_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    lasso = Lasso(**model_params)
    lasso.fit(X_train, Y_train)
    return lasso