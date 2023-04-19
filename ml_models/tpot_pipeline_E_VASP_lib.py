import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

def model2():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, min_samples_split=4)),
	    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
	    XGBRegressor(learning_rate=0.1, max_depth=7, min_child_weight=1, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.8500000000000001, verbosity=0)
	)
	
	return exported_pipeline