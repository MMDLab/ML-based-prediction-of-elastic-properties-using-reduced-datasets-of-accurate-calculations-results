import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor

def model2():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=XGBRegressor(learning_rate=1.0, max_depth=2, min_child_weight=2, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.15000000000000002, verbosity=0)),
	    SelectFwe(score_func=f_regression, alpha=0.023),
	    ZeroCount(),
	    LassoLarsCV(normalize=False)
	)
	
	return exported_pipeline