import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator

def model1():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=RidgeCV()),
	    RobustScaler(),
	    ExtraTreesRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
	)
	
	return exported_pipeline