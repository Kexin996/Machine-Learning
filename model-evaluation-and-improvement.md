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
```

* benefits:
  * more generalize
  * tell us how sensitive our models are to training datasets
    * **high variance ---> high sensitivity**
  *   we can use more data to fit the models

      * e.g. normal: 75% training and 25% testing
      * 10-fold: 90% training and 10% testing

      \
