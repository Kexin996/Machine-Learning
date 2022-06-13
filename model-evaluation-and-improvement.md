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



#### Search over spaces that are not grids

* sometimes trying the combination of all the parameters can be different
  * e.g. SVC ---> different kernels, different

```
# we make it a list of dictionaries
param_grid = [{'kernel': ['rbf'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
                  {'kernel': ['linear'],
'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

```



#### Using different cross-validation strategies with grid search

* by default, the grid-validation uses **stratified k-fold cross validation**
* we can change the parameter 'cv'



#### Nested Cross Validation

* reason:
  * we still need to do do a single split of test and train data
* nested cross validation: **multiple split ---> two loops**
  * outer loop: splitting the training and testing sets
  * inner loop: running a grid search
* return a list of scores ---> how well it can generalize
* **doesn't provide a model that can be used for new data**

```
# inplementation: 
# we just add one instance
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5),
                             iris.data, iris.target, cv=5)

# the inner cross-validation strategies don't have to be the same
# as the outer
```

* time consuming&#x20;



### \*Parallelizing cross-validation and grid search

* **building a model using a particular parameter setting or a particular cross validation split is independent from other parameters and models ---> parallelization over multiple CPU cores**

```
# we can do it by setting the n_jobs
n_jobs = -1 ---> using all available  CPU cores
```

* scikit-learn **does not allow nesting of parallel operations**.
  * we cannot use the same jobs in different operations
* p. 277



## Evaluation Metrics and Scoring

* business metric: **high-level goal**
* business impact: the **consequence of choosing a particular algorithm for the application**
* the metric should capture the business impact



### Metrics for Binary Classification

* accuracy:

![](<.gitbook/assets/image (1).png>)

* type I errors: false positive ---> no but shows yes
* type II errors: false negative ---> yes but shows no
* imbalanced datasets: in a data set, **one class is much more frequent than the others**
  * **the learning model is biased ---> the accuracy is fake**
  * e.g. the accuracy 90% from 90% data are true, 10% data are false may not be helpful
* the classification will always predict the majority, but how about the **costs**?&#x20;
  * **the cost for false positive and false negative are different**



### Confusion Matrix

* two x two matrix
* rows: true classes
* columns: predicted data

![](<.gitbook/assets/Screen Shot 2022-06-13 at 4.06.10 PM.png>)

```
from sklearn.metrics import confusion_matrix 
confusion = confusion_matrix(y_test, pred_logreg)
```

* it is manual and we need time to **summarize** it

#### Precision

* **positive predictive value** (PPV)
* when the goal is to limit the number of **false positive**

![](<.gitbook/assets/Screen Shot 2022-06-13 at 4.12.57 PM.png>)

#### Recall

* sensitivity, hit rate, or **true positive rate** (TPR)
* **how many positive samples are captured**
* when we want to **avoid** **false negative**

![](<.gitbook/assets/Screen Shot 2022-06-13 at 4.14.39 PM.png>)

* there is a trade off between **precision and recall**



#### F-score

* one way to summarize precision and recall
* more robust than accuracy for imbalanced dataset
* f1 score:

![](<.gitbook/assets/Screen Shot 2022-06-13 at 4.24.37 PM.png>)

```
from sklearn.metrics import f1_score

score1 = f1_score(y_test, pred_most_frequent)

```

* disadvantage: **harder to interpret and explain**
* more comprehensive summary:

```
from sklearn.metrics import classification_report

print(classification_report(y_test, pred_most_frequent,
                                target_names=["not nine", "nine"]))
```

![support: the number of samples to each class](<.gitbook/assets/Screen Shot 2022-06-13 at 4.29.18 PM.png>)

* how to get the 'avg':
  * (each number in the column except for the support, working as weight, \* the corresponding class value in support) / total
* we need to **look at all the values inside the report together to differentiate a dummy bad model and a good model.**



