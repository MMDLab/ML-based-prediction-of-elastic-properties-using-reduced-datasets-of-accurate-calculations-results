import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFwe, VarianceThreshold, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
def model1():
	exported_pipeline = make_pipeline(
	    SelectFwe(score_func=f_regression, alpha=0.044),
	    VarianceThreshold(threshold=0.1),
	    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=1, min_samples_split=11, n_estimators=100)),
	    ElasticNetCV(l1_ratio=1.0, tol=1e-05)
	)
	
	return exported_pipeline