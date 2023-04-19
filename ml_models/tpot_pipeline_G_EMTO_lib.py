import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
def model1():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=RidgeCV()),
	    SelectPercentile(score_func=f_regression, percentile=76),
	    ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
	)
	
	return exported_pipeline