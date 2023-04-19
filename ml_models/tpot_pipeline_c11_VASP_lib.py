import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
def model1():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.4, min_samples_leaf=5, min_samples_split=7, n_estimators=100)),
	    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=6, min_samples_split=8, n_estimators=100)),
	    SelectFwe(score_func=f_regression, alpha=0.016),
	    LinearSVR(C=20.0, dual=False, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=1e-05)
	)
	
	return exported_pipeline