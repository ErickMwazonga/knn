import numpy as np
import pandas as pd
from collections import defaultdict
import numbers

def calculate_weighted_hamming(data):
    categories_dist = []
    
    for category in data:
        X = pd.get_dummies(data[category])
        X_mean = X * X.mean()
        X_dot = X_mean.dot(X.transpose())
        X_np = np.asarray(X_dot.replace(0,1,inplace=False))
        categories_dist.append(X_np)

    categories_dist = np.array(categories_dist)
    distances = hmean(categories_dist, axis=0)

    return distances
  