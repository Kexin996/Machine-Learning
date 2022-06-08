# Unsupervised Learning and Preprocessing

* no predefined class
* use:
  * exploratory setting
  * preprocessing



## Preprocessing

```
StandardScaler
# each feature's mean is 0 and variance is 1
# disadvantage: the outliers may influence it

RobustScaler
# same scale as standardscaler, but uses median and quartiles
# but it ignores outliers

MinMaxScaler
# shift the data such that all features are exactly 0 between and 1
# since the min becomes 0, and the max becomes 1
# we need to transform both the train and test sets


Normalizer
# each feature vector's length is 1
# use only the direction of the data matters

```

* why sometimes the scaled test set are not between 1 and 0:
  * transformer applies the same transform to the training and test set
    * _**x\_after\_scaled = x\_test\_value - training\_set\_minimum / training\_set\_range、**_
    * _**same transformation, so for the training set, the max = 1 / min = 0**_
* we need to apply the **same transformation** to both sets
  * don't scale them separately
* shortcut:

![](<.gitbook/assets/Screen Shot 2022-06-07 at 4.15.59 PM.png>)

## Dimensionality Reduction

### Principal Component Analysis (PCA)

* rotates the dataset so that the features re statistically uncorrelated
* procedures:
  * find the line where the features are most correlated with each other
    * max variance
  * using vector multiplication to rotate it
    * cos90 = 1, which will contain the most information

![\*the direction doesn't matter](<.gitbook/assets/Screen Shot 2022-06-07 at 4.33.19 PM.png>)

* so now the two features are not correlated
* the mean was subtracted from the data between rotation

![](<.gitbook/assets/Screen Shot 2022-06-07 at 4.37.21 PM.png>)

* every time find the most 'fit' line through the origin (to maximize SS(distance)) <---Eigenvalues
* it is comparable to that we use the new PCAs to predict
* overlapping data points: not that informative

![](<.gitbook/assets/Screen Shot 2022-06-07 at 5.51.39 PM.png>)

* it is unsupervised but it can be used to check the dimension, something related to the algorithms that we can use
