from sklearn.model_selection import GridSearchCV


def cv(X_train, y_train, clf, parameters, model_name):
    model = clf
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring="r2", cv=5)
    grid_search.fit(X_train, y_train)
    print(model_name)
    best_params = grid_search.best_params_
    return best_params, grid_search.best_estimator_