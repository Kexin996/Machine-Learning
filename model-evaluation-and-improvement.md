# Model Evaluation and Improvement

* **how well our models can fit the new data, not training data**

## Cross Evaluation

* most common one: **k-fold cross-validation**
  * k: the number of cross-validation we want
    * and the training is divided into k-partitions
    * default: 3
    * **random split**
  * procedures:
    * instead of splitting testing and training data once, we **do it k times**
    * first kth becomes test, and the following becomes training data (**random**)
    * ... repeat it for k times
  * to summarize the result, we takes the **average**

```
from sklearn.model_selection import cross_val_score 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target)

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
# cv: changes the number of folds
# if cv is a number, it is Stratified K-fold Cross-Validation
# if it is KFold(n_splits=3), it is standard cross-validation
```

* benefits:
  * more generalize
  * tell us how sensitive our models are to training datasets
    * **high variance ---> high sensitivity**
  * we can use more data to fit the models
    * e.g. normal: 75% training and 25% testing
    * 10-fold: 90% training and 10% testing
* disadvantage: **high computation costs**
* scikit-learn uses it as **default**
*

### Stratified K-fold Cross-Validation

* ensure the **same proportions** of data
  * **each fold ensures the same proportion as the data**&#x20;

### Others

* Leave-one-out cross-validation:
  * only **a single element** is the test element
  * small datasets

```
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
```

* Shuffle-split cross-validation
  * the data is split into train\__size train set, and test\_size test_&#x20;
  * repeats for **n\_tier** times
  * it also has stratified version

```
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
```
