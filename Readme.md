## KNN Project

---

Students

1. Mwandeje E. Mwazonga - 111778
2. William Kamau – 072291

---

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
   `Remove columns which according to me do not make any contribution to our prediction.`

2. Calculate Hamming distance on categorical values.
   `For one variable, it is equal to 1 if the values between point A and point B are different, else it is equal the relative frequency of the distribution of the value across the variable. For multiple variables, the harmonic mean is computed up to a constant factor.`

3. Infer distance by Computing global mean for numeric column and global mode for categorical ones.
   Compute the pairwise distance attribute by attribute in order to account for different variables type:

   - Continuous
   - Categorical
     For ordinal values, provide a numerical representation taking the order into account.
     Categorical variables are transformed into a set of binary ones.
     If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric variables are all normalized in the process.
     If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.

4. Apply the imputation strategy to get the distance using KNN implementation
   Replace the missing values within the target variable based on its k nearest neighbors identified with the
   attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
   remains missing. If there is a problem in the parameters provided, returns None.
   If to many neighbors also have missing values, leave the missing value of interest unchanged.

### Testing the Model

1. Download the [mini-project](https://github.com/ErickMwazonga/knn.git) from my github repo.
2. Go into the project repo
   `cd knn`
3. Install required packages
   `pip install -r requirements.txt`
