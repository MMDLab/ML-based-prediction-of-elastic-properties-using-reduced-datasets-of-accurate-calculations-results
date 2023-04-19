import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
def model1():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=7, max_features=0.7000000000000001, min_samples_leaf=4, min_samples_split=3, n_estimators=100, subsample=0.45)),
	    RidgeCV()
	)
	
	return exported_pipeline