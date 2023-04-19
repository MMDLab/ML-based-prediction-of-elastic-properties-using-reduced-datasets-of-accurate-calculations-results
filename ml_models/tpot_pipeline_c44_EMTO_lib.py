import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
def model1():
	exported_pipeline = make_pipeline(
	    StandardScaler(),
	    StackingEstimator(estimator=LinearSVR(C=0.001, dual=True, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.001)),
	    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=1.0, learning_rate="invscaling", loss="huber", penalty="elasticnet", power_t=0.5)),
	    RandomForestRegressor(bootstrap=False, max_features=0.15000000000000002, min_samples_leaf=1, min_samples_split=6, n_estimators=100)
	)
	
	return exported_pipeline