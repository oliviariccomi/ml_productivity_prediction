from sklearn.feature_selection import VarianceThreshold
import statistics


def feature_selection(X_train, X_test):
    for i in X_train.columns:
        print("Variance of " + i + ": " + str(statistics.variance(X_train[i])))
    # Removing year because has 0 variance
    # Removing day because of high correlation with quarter (risking multicollinearity)
    # NB) targeted_productivity also has low variance but high correlation with target variability
    X_train.drop(['year', 'day'], axis=1, inplace=True)
    X_test.drop(['year', 'day'], axis=1, inplace=True)

    return X_train, X_test
