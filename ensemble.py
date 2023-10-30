from sklearn.ensemble import BaggingRegressor


def ensemble_bagging(model, SEED, X_train, y_train, X_test):
    clf = BaggingRegressor(base_estimator=model,n_estimators=10, random_state=SEED)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction
