import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

import cross_validation
import eda
import preprocessing
import evaluation
import ensemble

d = pd.read_csv('./data/garments_worker_productivity.csv')

SEED = 126

# models = ['KNearestNeighbors', 'SupportVectorRegressor', 'RidgeRegressio '}
# declared two versions of models since I want to test on both preprocessed and non preprocesed data
knn = KNeighborsRegressor()
knn_preprocessed = KNeighborsRegressor()
svr = SVR()
svr_preprocessed = SVR()
ridge = Ridge()
ridge_preprocessed = Ridge()


########################
#        EDA           #
########################

df = eda.eda_analysis(d)

##########################
#    SPLITTING DATASET   #
##########################

X = df.drop(columns="actual_productivity")
y = df.actual_productivity.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
# Copy to make sure
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

############################################
#    PREPROCESSING & FEATURE SELECTION     #
############################################

X_train_preprocessed, X_test_preprocessed = preprocessing.feature_selection(X_train, X_test)

#  Normalization
scaler = MinMaxScaler()

# Scaling no processed data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Scaling preprocessed data
X_train_scaled_preprocessed = scaler.fit_transform(X_train_preprocessed)
X_test_scaled_preprocessed = scaler.transform(X_test_preprocessed)

###########################
#    CROSS VALIDATION     #
###########################

## KNN
hparameters_knn = {"n_neighbors": list(range(1, 16)), "weights": ["uniform", "distance"]}

# Non preprocessed data
tbegin = time()
BEST_PARAM_KNN, best_estimator_knn = cross_validation.cv(X_train, y_train, knn, hparameters_knn, "KNN for non preprocessed data")
tend = time()
print("Timing: ", tend - tbegin)
print(BEST_PARAM_KNN, best_estimator_knn)

# Preprocessed data
tbegin = time()
BEST_PARAM_KNN_PREP, best_estimator_knn_prep = cross_validation.cv(X_train_scaled_preprocessed, y_train, knn_preprocessed, hparameters_knn, "KNN for preprocessed data")
tend = time()
print("Timing: ", tend - tbegin)
print(BEST_PARAM_KNN_PREP, best_estimator_knn_prep)


## SVR
hparameters_svr = {"C": [0.1, 1, 10], "epsilon": [0.01, 0.1, 0.2, 0.5, 1, 2], "kernel": ["poly", "linear", "rbf"]}

# Non preprocessed data but using scaled because it would take too much time otherwise
tbegin = time()
BEST_PARAM_SVR, best_estimator_svr = cross_validation.cv(X_train_scaled, y_train, svr, hparameters_svr,
                                                                         "SVR for non preprocessed data")
tend = time()
print("Timing: ", tend - tbegin)
print(BEST_PARAM_SVR, best_estimator_svr)

# Preprocessed data
tbegin = time()
BEST_PARAM_SVR_PREP, best_estimator_svr_prep = cross_validation.cv(X_train_scaled_preprocessed, y_train, svr_preprocessed, hparameters_svr, "SVR for preprocessed data")
tend = time()
print("Timing: ", tend - tbegin)
print(BEST_PARAM_SVR_PREP, best_estimator_svr_prep)


## RIDGE
hparameters_ridge = {"alpha": [0.001, 0.01, 0.1, 0.5, 1, 10, 100]}

# Non preprocessed data
tbegin = time()
BEST_PARAM_RIDGE, best_estimator_ridge = cross_validation.cv(X_train, y_train, ridge, hparameters_ridge, "RIDGE for non preprocessed data")
tend = time()
print("Timing: ", tend - tbegin)
print(BEST_PARAM_RIDGE, best_estimator_ridge)

# Preprocessed data
tbegin = time()
BEST_PARAM_RIDGE_PREP, best_estimator_ridge_prep = cross_validation.cv(X_train_scaled_preprocessed, y_train, ridge_preprocessed, hparameters_ridge, "RIDGE for preprocessed data")
tend = time()
print("Timing: ", tend - tbegin)
print(BEST_PARAM_RIDGE_PREP, best_estimator_ridge_prep)



#############################
#    MODEL WITH BEST HP     #
#############################
models = ["KNN NO PREPROCESSING", "KNN WITH PREPROCESSING", "SVR NO PREPROCESSING", "SVR WITH PREPROCESSING",
          "RIDGE NO PREPROCESSING", "RIDGE WITH PREPROCESSING"]

knn_model_final = best_estimator_knn
knn_model_preprocessed_final = best_estimator_knn_prep
svr_model_final = best_estimator_svr
svr_model_preprocessed_final = best_estimator_svr_prep
ridge_model_final = best_estimator_ridge
ridge_model_preprocessed_final = best_estimator_ridge_prep


#############################
#       PREDICTION          #
#############################
y_predicted = []

# KNN
tbegin = time()
y_knn = knn_model_final.predict(X_test)
y_predicted.append(y_knn)
tend = time()
print("Process time for KNN: ", tend - tbegin)

tbegin = time()
y_preprocessed_knn = knn_model_preprocessed_final.predict(X_test_scaled_preprocessed)
y_predicted.append(y_preprocessed_knn)
tend = time()
print("Process time for KNN with preprocessed data: ", tend - tbegin)

# SVR
tbegin = time()
y_svr = svr_model_final.predict(X_test)
y_predicted.append(y_svr)
tend = time()
print("Process time for SVR: ", tend - tbegin)

tbegin = time()
y_preprocessed_svr = svr_model_preprocessed_final.predict(X_test_scaled_preprocessed)
y_predicted.append(y_preprocessed_svr)
tend = time()
print("Process time for SVR Preprocessed: ", tend - tbegin)

# RIDGE
tbegin = time()
y_ridge = ridge_model_final.predict(X_test)
y_predicted.append(y_ridge)
tend = time()
print("Process time for RIDGE: ", tend - tbegin)

tbegin = time()
y_preprocessed_ridge = ridge_model_preprocessed_final.predict(X_test_scaled_preprocessed)
y_predicted.append(y_preprocessed_ridge )
tend = time()
print("Process time for RIDGE Preprocessed: ", tend - tbegin)


#############################
#       EVALUATION          #
#############################

evaluation.evaluate(y_test, y_predicted, models)


########################
#       ENSEMBLE       #
########################
y_pred_knn_bagging = ensemble.ensemble_bagging(knn_model_preprocessed_final, SEED, X_train_scaled_preprocessed, y_train, X_test_scaled_preprocessed)
print('\nKNN Bagging R^2 score: ', r2_score(y_test, y_pred_knn_bagging))
y_pred_svr_bagging = ensemble.ensemble_bagging(svr_model_preprocessed_final, SEED, X_train_scaled_preprocessed, y_train, X_test_scaled_preprocessed)
print('\nSVR Bagging R^2 score: ', r2_score(y_test, y_pred_svr_bagging))
y_pred_ridge_bagging = ensemble.ensemble_bagging(ridge_model_final, SEED, X_train_scaled, y_train, X_test_scaled)
print('\nRidge Bagging R^2 score is: ', r2_score(y_test, y_pred_ridge_bagging))

