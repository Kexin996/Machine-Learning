# Supervised Machine Learning

## KNN Algorithm

* easy to understand + implement
* baseline
* more neighbors, more simple&#x20;
* inefficient when the training set is large / too many features / many missing features (NaN)



## Linear Models

* single feature&#x20;

### linear regression (ordinary least squares)

* overfitting for high-dimension dataset



### ridge regression (avoid overfitting)

* we want to keep the coefficients as small as possible, but at the same time predict test\_data well
* because we want to decrease the influences **brought by each feature**
* how to prevent overfit: increase alpha
* we need&#x20;
* enough data ---> ridge and linear will have the same r^2
* L2



### lasso regression

* ignore some features ---> some coefficients = 0
* how to prevent underfit: decrease alpha
* easy to interpret + find the most important features
* L1



### logistic regression

* default: L2 regularization
* C: how much do you trust the training set
* high C: more complex
* low C: less complex
* multi-class classification



### linear models for multi-class classification

#### 1. one-vs.-rest

* extend a binary classification to multi-class classification



## Non-Linear

### Bayesian algorithm&#x20;

* gaussian - normal distribution
* b - binary, occurence
* multi - frequency

### Decision Tree

* we can decrease overfitting
* pre-pruning:
  * max depth
  * max leaf nodes
  * min samples leaf&#x20;
* prediction is useless outside training data
* easy to understand/visualize + no need for standardization&#x20;

#### Ensembles of Decision Trees

1. random forests

* building many decision trees and taking their average
* bootstrap
* better accuracy than linear models / single tree without any tuning
  * default works well
* improve the speed: use more CPU-cores
* doesn't work well for high dimensional / sparse data
  * e.g. text data
* use more memory / time to train
* classification: _max\_features = sqrt(n\__features) (smaller features decrease overfitting)
* regression: _max\_features = n\_features_



2\. gradient boosted regression trees

* both regression and classification
* shallow (usually max\_depths <= 5) + no randomization + sensitive to parameter and strong pruning
* combine simple models (weak learners)
* learning\_rate: how strongly each tree corrects the mistakes of previous trees
  * higher rate: more complex models
  * low learning rate ---> more trees needed
* careful tuning + long time to train
* more trees ---> overfitting (it is not random)
* fit n\_estimators depends on time and memory budget, then tune learning rates
* XGB: extreme gradient boosted regression trees
  * use both L1 and L2 to regularize

**first try random forest, then gradient boosted regression trees**

****

### **Kernelized Support Vector Machine (SVMs)**

* useful for those models cannot be defined as a hyperplane
* often classification
* for 2D systems, we can add one more feature to make it 3D and the plane is not linear anymore
  * ?: which feature to add
  * solution: kernel trick
    * radial basis kernel (Gaussian kernel):
  * ![](<.gitbook/assets/Screen Shot 2022-06-06 at 3.33.55 PM.png>)
    * polynomial trick: computes all possible polynomial up to a certain degree of original features
* support vector: the points on the boundary
  * classification decision: distance to support vector
  * gamma: controls the width of kernels
    * low gamma: low complexity
    * high gamma: high complexity
    * default gamma: 1/n\_samples
* required all the features to vary on a similar scale
  * rescaling:
    * _(x\_train - x\_min) / range_
* time+memory usage / needs preprocessing + tuning parameter / hard to inspect



### Neural Networks (Deep Learning)

* aka multilayer perceptrons (MLPs)
* when we cannot make a linear separation&#x20;
* &#x20;a kind of blackbox&#x20;
  * we actually don't know how the hidden layers work
* for each layer, the input will be calculated by a matrix to get a sum, and the sum will be inputed into a non-linear function
  * relu (rectifying nonlinearity) - default
    * rule off the negative value
  * tangens hyperbola (tanh)
    * change between -1 and 1

![](<.gitbook/assets/Screen Shot 2022-06-07 at 10.26.00 AM.png>)

* &#x20;complexity:
  * more hidden nodes
  * more hidden layer
  * tanh nonlinearity
* control complexity:&#x20;
  * L2
  * alpha is quite low by default
* the value needs to vary in the same scale - rescaling
  * _(x\_train-x\_mean) / std\_on\_train_
* weights are set randomly&#x20;
  * same inputs but different seeds - different models
* advantages
  * large data and complex models
* disadvantages
  * long time to train
  * data needs to be homogenous - all the features have the similar meaning
  * need to be tuned
* how to choose number of layers and nodes:
  * first create a overfitting model
  * then regularize it
* algorithms for learning the parameter:
  * **adam** (default)
    * &#x20;quite sensitive to scaling
  * **lbfgs**&#x20;
    * robust but takes a long time for large models / large dataset
  * sgd
    * need to tune many parameters



## Uncertainty Estimates

* how certain the point is in the class



### Decision Function (Binary)

```
decision_function()
# its shape: (n,)
# positive number: belongs to this class
# negative value: belongs to other classes

# positive: second class
# negative first class
# since we will turn it into true/false
# need scaling
```

* however, it is hard to make out the boundary between two classes



### Predicting Probabilities (Binary)

```
predict_proba()
# its shape: (n,2) 
# for binary classifications
```

* calibrated: certainty = correct prediction
* easier to make out boundary



### Uncertainty in multi-class classification

### Decision Function (Multi-class)

* shape: (n\__samples, n\__classes)
* large score: more likely



### Predicting Probabilities (Multi-class)

* same as binary one&#x20;

how to recover prediction:&#x20;

```
np.argmax() # returns the index of the largest value of each row
```

![](<.gitbook/assets/Screen Shot 2022-06-07 at 2.58.12 PM.png>)

