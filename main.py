import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
import numbers

from knn import knn_implementation

def main():
    df = pd.read_csv("titanic_dataset.csv") 

    knn = knn_implementation(
        target=df['Age'],
        attributes=df.drop(['Age', 'PassengerId'], 1),
        aggregation_method="median",
        k_neighbors=10,
        numeric_distance='euclidean',
        categorical_distance='hamming',
        missing_neighbors_threshold=0.8
    )

    return knn

if __name__ == '__main__':
    main()
