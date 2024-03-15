import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import warnings
import time
import sys
import os
from tqdm.notebook import tqdm

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import RepeatedKFold, GridSearchCV, ParameterGrid
from sklearn.utils import Bunch, check_array
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, pairwise_kernels

if os.path.dirname(os.getcwd()) not in sys.path:
    sys.path.append(os.path.dirname(os.getcwd()))
from rcca import TwoViewRCCA

rs_num = 42
rng = np.random.default_rng(rs_num)

n_splits = 3
n_repeats = 20

sample_num = 100
X_dimension_num = 30
Y_dimension_num = 10

# construct X
X = rng.normal(loc=0.5, scale=1.0, size=(sample_num, X_dimension_num))

# construct noise
delta_1 = rng.normal(loc=0.0, scale=0.2, size=(sample_num, ))
delta_2 = rng.normal(loc=0.0, scale=0.1, size=(sample_num, ))
delta_3 = rng.normal(loc=0.0, scale=0.1, size=(sample_num, ))

# construct Y
Y = rng.normal(loc=0.1, scale=1.0, size=(sample_num, Y_dimension_num))
Y[:, 0] = X[:, 3] + delta_1
Y[:, 1] = X[:, 0] + delta_2
Y[:, 2] = X[:, 1] + delta_3

# split dataset into training and testing
[X_train, X_test, Y_train, Y_test] = train_test_split(
    X, Y, test_size=0.20, random_state=rs_num
)

test_pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('rcca', TwoViewRCCA(n_components=1, solving_method='gen_eig', standardization=True, random_state=rs_num)),
        ('lr', LinearRegression())
    ]
)

test_pipeline.fit(X_train, Y_train)
