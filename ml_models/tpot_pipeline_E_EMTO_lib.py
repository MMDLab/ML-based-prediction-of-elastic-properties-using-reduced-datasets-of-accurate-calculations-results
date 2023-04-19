import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator, ZeroCount
def model1():
	exported_pipeline = make_pipeline(
	    ZeroCount(),
	    StackingEstimator(estimator=RidgeCV()),
	    ZeroCount(),
	    ZeroCount(),
	    StandardScaler(),
	    RandomForestRegressor(bootstrap=False, max_features=0.2, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
	)
	
	return exported_pipeline