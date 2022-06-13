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



### Stratified K-fold Cross-Validation

* ensure the **same proportions** of data
  * **each fold ensures the same proportion as the data**&#x20;

### Others

* **Leave-one-out cross-validation**
  * only **a single element** is the test element
  * small datasets

```
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
```

* **Shuffle-split cross-validation**
  * the data is split into train\__size train set, and test\_size test_&#x20;
  * repeats for **n\_tier** times
  * it also has stratified version

```
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
```

* **GroupKfold**
  * the group indicates the groups that **cannot be split into both train\_**_**set and test\_set**_

```
from sklearn.model_selection import GroupKFold

# create synthetic dataset
X, y = make_blobs(n_samples=12, random_state=0)

# assume the first three samples belong to the same group, 
# then the next four, etc. 
groups=[0,0,0,1,1,1,1,2,2,3,3,3]

scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
```

![](<.gitbook/assets/Screen Shot 2022-06-13 at 1.48.16 PM.png>)



## Grid Search

* improve the generalization of models
* most common
* logic:
  * we cannot use test data to "tune" the parameters
  * we need three sets now:
  * we train on both **training and validation** set

![](<.gitbook/assets/Screen Shot 2022-06-13 at 1.54.25 PM.png>)

![the training of validation set is not to actually train it, we just use it to fit and tune](<.gitbook/assets/Screen Shot 2022-06-13 at 1.56.02 PM.png>)

## Grid Search with Cross-Validation

* takes a long time ---> train more models, due to the parameter c'cv'

![](<.gitbook/assets/Screen Shot 2022-06-13 at 2.00.45 PM.png>)

* it is quite similar to: **we use cross validation on the training set to split into the 'real' test set and development set**
* how to use it:

```
# we need to create a parameter grid in dictionary form
# indicate the range we want
# ex:
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# it will use cross-validation instead of having three sets
# but we still need the train set and test set
X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0)

grid_search.fit(X_train, y_train)
# it will use SCV() to predict
# automatically uses the best parameters to fit the data

grid_search.best_params_
# tell us the parameters it chooses

grid_search.best_score_
# best cross-validation accuracy (the mean score over the different splits
# while using the best parameters)

grid_search.best_estimator_
# access the attributes of the model

grid_search.cv_results_
# it returns a dictionary
# in order to see it, we need to turn it into a dataframe
results = pd.DataFrame(grid_search.cv_results_)
```

* best\__score_\_: the best score while performing cross validation on training set
* score: the score we get from using the whole training set
* we can use heatmap to see the sensitivity of the mode with respect to parameters
* no change in accuracy for different parameter in the range&#x20;
  * \---> **wrong range of parameter** we choose
  * \---> the feature may be **trivial**
  * so it is good to try **extreme points**&#x20;
* vertical stripe pattern ---> maybe **only one feature** matters



####

