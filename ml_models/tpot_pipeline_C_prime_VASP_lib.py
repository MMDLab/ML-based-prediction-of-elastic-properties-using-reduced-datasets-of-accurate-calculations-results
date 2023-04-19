import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFwe, VarianceThreshold, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount

def model2():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=37, p=2, weights="uniform")),
	    SelectFwe(score_func=f_regression, alpha=0.004),
	    VarianceThreshold(threshold=0.0001),
	    ZeroCount(),
	    ZeroCount(),
	    ZeroCount(),
	    PCA(iterated_power=10, svd_solver="randomized"),
	    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.3, min_samples_leaf=16, min_samples_split=7, n_estimators=100)),
	    RidgeCV()
	)
	
	return exported_pipeline