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

1. Remove Irrelevant columns
   Remove columns which according to me do not make any contribution to our prediction.

2. Compute global mean for numeric column and global mode for categorical ones.

3. Apply the knn_impute function

4. Build the predictive model

### Testing the Model

1. Download the [mini-project](https://github.com/ErickMwazonga/knn.git) from my github repo.
2. Go into the project repo
   `cd knn`
3. Install required packages
   `pip install -r requirements.txt`
