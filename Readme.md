## Handle Missing Values using Weighted Hamming Distance

### Problem

Missing values come in different shapes in a dataset,
one will encounter missing and null values. A crucial point is how to deal with them.
There are many strategies to handle them for KNN implementation.

#### My Proposal

Approximate the missing values based on the distances of the points closest to the missing point.

### Dataset

Use the [Titanic Dataset](https://www.kaggle.com/c/titanic/data) to Predict which passenger survived the Titanic.

### Implementation Algorithm
