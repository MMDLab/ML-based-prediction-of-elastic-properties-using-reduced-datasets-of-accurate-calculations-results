import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

def model2():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=6, min_samples_split=19, n_estimators=100)),
	    SelectPercentile(score_func=f_regression, percentile=16),
	    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.1, min_samples_leaf=11, min_samples_split=9, n_estimators=100)),
	    RidgeCV()
	)
	
	return exported_pipeline