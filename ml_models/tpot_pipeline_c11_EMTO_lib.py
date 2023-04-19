import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
def model1():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=1.0, fit_intercept=True, l1_ratio=0.0, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=0.5)),
	    StackingEstimator(estimator=RidgeCV()),
	    RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=2, min_samples_split=4, n_estimators=100)
	)
	
	return exported_pipeline