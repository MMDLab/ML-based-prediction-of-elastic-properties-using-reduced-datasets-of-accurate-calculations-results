import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
def model1():
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
	    SelectPercentile(score_func=f_regression, percentile=32),
	    ElasticNetCV(l1_ratio=0.30000000000000004, tol=1e-05)
	)
	
	return exported_pipeline