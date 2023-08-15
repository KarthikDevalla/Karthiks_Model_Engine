# fluid_domain.art import KMEngine

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
---
## class KMEngine(data, column, test_ratio=0.2)
---

KMEngine (Karthik's Model Engine) is a versatile tool designed to streamline machine learning model training, evaluation, and customization. With support for both regression and classification tasks, Kmodel_engine offers an automated workflow that includes data preprocessing, feature selection, exploratory data analysis (EDA), model training, and metric evaluation. This class integrates an array of popular machine learning models from libraries like Scikit-Learn, XGBoost, and LightGBM. Additionally, it facilitates model parameter customization, empowering users to efficiently build, assess, and refine machine learning models tailored to their specific datasets and tasks. As of now this class cannot remove outliers, to remove them use [O_sieve](https://pypi.org/project/vcosmos/) or [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)


---
## Features
- Supports both regression and classification tasks
- Integrates a wide range of machine learning models from popular libraries such as Scikit-Learn, XGBoost, LightGBM, and more
- Performs automated data preprocessing including missing value imputation, scaling, and one-hot encoding
- Allows for exploratory data analysis (EDA) through YData Profiling
- Provides various evaluation metrics including accuracy, F1-score, precision, recall, ROC AUC, R-squared, adjusted R-squared, mean absolute error (MAE), and root mean squared error (RMSE)
- Supports custom parameter tuning for models
---

## Parameters:

- data: dataframe
    - The __data__ on which the evalution of models should occur.
  
- column: str
    - __Target column__, the dependent variable.

- test_ratio: float, default=0.2
    - The conventional training and testing split ratio.

---
## Installation

```shell
pip install KMEngine
```

## Usage


### EDA (EXtrapolatory Data Analysis)
```python
import pandas as pd
from fluid_domain.art import KMEngine
# Reading a dataset using pandas.
df=pd.read_csv('tested.csv')
engine=KMEngine(df,'Survived')
eda=engine.EDA()
# EDA done. Check your working directory for the html file. 
# Produces an html file with the name 'KME_data_report.html' in the working directory.
```

### Classification:
```python
import pandas as pd
from fluid_domain.art import KMEngine
# Reading a dataset using pandas.
df=pd.read_csv('tested.csv')
engine=KMEngine(df,'Survived')
result=engine.super_learning()
print(result)

# Engine Summoned.
# Loaded Data Successfully with 418 rows and 12 columns.
# Building models and Training them. This might take a while...
# Engine Encountered Discrete Data. Hence, proceeding with Classification

# Writing models to respective keys: ['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier', 'Xtreme Gradient Boosting Classifier', 'Stochastic Gradient Descent Classifier', 'Gradient Boosting Classifier', 'Adaptive Boost Classifier', 'Light Gradient Boosting Classifier', 'Extra Trees Classifier', 'Support Vector Classification', 'K Nearest Neighbors Classifier', 'Ridge Classifier', 'MLP Classifier', 'Quadratic Discriminant Analysis', 'Linear Discriminant Analysis', 'Naive Bayes Classifier']

# Currently Running : Logistic Regression
# Currently Running : Random Forest Classifier
# Currently Running : Decision Tree Classifier
# Currently Running : Xtreme Gradient Boosting Classifier
# Currently Running : Stochastic Gradient Descent Classifier
# Currently Running : Gradient Boosting Classifier
# Currently Running : Adaptive Boost Classifier
# Currently Running : Light Gradient Boosting Classifier
# Currently Running : Extra Trees Classifier
# Currently Running : Support Vector Classification
# Currently Running : K Nearest Neighbors Classifier
# Currently Running : Ridge Classifier
# Currently Running : MLP Classifier
# Currently Running : Quadratic Discriminant Analysis
# Currently Running : Linear Discriminant Analysis
# Currently Running : Naive Bayes Classifier

# All models evaluations:
# +----------------------------------------+----------+----------+-----------+--------+---------+
# |                 Model                  | Accuracy | F1-Score | Precision | Recall | ROC AUC |
# +----------------------------------------+----------+----------+-----------+--------+---------+
# |          Logistic Regression           |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |        Random Forest Classifier        |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |        Decision Tree Classifier        |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |  Xtreme Gradient Boosting Classifier   |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# | Stochastic Gradient Descent Classifier |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |      Gradient Boosting Classifier      |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |       Adaptive Boost Classifier        |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |   Light Gradient Boosting Classifier   |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |         Extra Trees Classifier         |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |     Support Vector Classification      |   0.98   |   0.97   |    0.97   |  0.97  |   0.97  |
# |     K Nearest Neighbors Classifier     |   0.99   |   0.98   |    0.97   |  1.0   |   0.99  |
# |            Ridge Classifier            |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |             MLP Classifier             |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |    Quadratic Discriminant Analysis     |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# |      Linear Discriminant Analysis      |   0.63   |   0.37   |    0.45   |  0.31  |   0.56  |
# |         Naive Bayes Classifier         |   1.0    |   1.0    |    1.0    |  1.0   |   1.0   |
# +----------------------------------------+----------+----------+-----------+--------+---------+
# Time Eaten :3.7687642574310303 secs

```
### Regression

```python
import pandas as pd
from fluid_domain.art import KMEngine
# Reading a dataset using pandas.
df=pd.read_csv('co2.csv')
engine=KMEngine(df,'CO2 Emissions(g/km)')
result=engine.super_learning()
print(result)

# Engine Summoned.
# Loaded Data Successfully with 7385 rows and 12 columns.
# Building models and Training them. This might take a while...
# Engine Encountered Continuous Data. Hence, proceeding with Regression

# Writing models to respective keys: ['Linear Regression', 'Random Forest Regression', 'Light Gradient Boosting Regressor', 'Xtreme Gradient Boosting', 'Decison Tree Regressor', 'Gradient Boosting Regressor', 'Adaptive Boosting Regressor', 'Stochastic Gradient Descent Regressor', 'Support Vector Regression', 'Extra Trees Regressor', 'Ridge Regression', 'Gamma Regressor', 'Huber Regressor', 'Poisson Regressor', 'Lasso Regressor', 'Elastic Net Regressor', 'K Nearest Neighbors Regressor', 'MLP Regressor']

# Currently Running : Linear Regression
# Currently Running : Random Forest Regression
# Currently Running : Light Gradient Boosting Regressor
# Currently Running : Xtreme Gradient Boosting
# Currently Running : Decison Tree Regressor
# Currently Running : Gradient Boosting Regressor
# Currently Running : Adaptive Boosting Regressor
# Currently Running : Stochastic Gradient Descent Regressor
# Currently Running : Support Vector Regression
# Currently Running : Extra Trees Regressor
# Currently Running : Ridge Regression
# Currently Running : Gamma Regressor
# Currently Running : Huber Regressor
# Currently Running : Poisson Regressor
# Currently Running : Lasso Regressor
# Currently Running : Elastic Net Regressor
# Currently Running : K Nearest Neighbors Regressor
# Currently Running : MLP Regressor

# All models evaluations:
# +---------------------------------------+----------------+-------------------+------------+------------+
# |                 Model                 |    R2 Score    | Adjusted R2 Score |    MAE     |    RMSE    |
# +---------------------------------------+----------------+-------------------+------------+------------+
# |           Linear Regression           |      0.89      |        0.89       |   12.25    |   19.71    |
# |        Random Forest Regression       |      0.99      |        0.99       |    2.57    |    6.3     |
# |   Light Gradient Boosting Regressor   |      0.97      |        0.97       |    4.46    |   10.14    |
# |        Xtreme Gradient Boosting       |      0.99      |        0.99       |    2.91    |    6.95    |
# |         Decison Tree Regressor        |      0.98      |        0.98       |    2.48    |    8.39    |
# |      Gradient Boosting Regressor      |      0.97      |        0.97       |    5.53    |   10.53    |
# |      Adaptive Boosting Regressor      |      0.88      |        0.88       |   15.38    |   20.93    |
# | Stochastic Gradient Descent Regressor | -1490880552.75 |   -1493104842.2   | 1606127.27 | 2247205.34 |
# |       Support Vector Regression       |      0.89      |        0.89       |    9.52    |   19.32    |
# |         Extra Trees Regressor         |      0.99      |        0.99       |    2.16    |    6.18    |
# |            Ridge Regression           |      0.89      |        0.89       |   11.74    |   18.88    |
# |            Gamma Regressor            |      0.89      |        0.89       |   12.05    |   20.65    |
# |            Huber Regressor            |      0.82      |        0.82       |    8.17    |   25.38    |
# |           Poisson Regressor           |      0.92      |        0.92       |   10.35    |    17.1    |
# |            Lasso Regressor            |      0.9       |        0.9        |   12.04    |   18.94    |
# |         Elastic Net Regressor         |      0.89      |        0.89       |   11.73    |   19.41    |
# |     K Nearest Neighbors Regressor     |      0.98      |        0.98       |    3.49    |    8.15    |
# |             MLP Regressor             |      0.91      |        0.91       |    9.01    |   17.89    |
# +---------------------------------------+----------------+-------------------+------------+------------+
# Time Eaten :23.698055028915405 secs

```
### Setting custom parameters

```python
import pandas as pd
from fluid_domain.art import KMEngine
# Reading a dataset using pandas.
df=pd.read_csv('tested.csv')
engine=KMEngine(df,'Survived')
custom_model = engine.set_custom_params('Random Forest Classifier', n_estimators=500, max_depth=10, random_state=666)
print(custom_model)

# Engine Summoned.
# Loaded Data Successfully with 418 rows and 12 columns.
# Custom parameters applied for Random Forest Classifier:
# {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 500, 'n_jobs': None, 'oob_score': False, 'random_state': 666, 'verbose': 0, 'warm_start': False}
# Currently Running : Random Forest Classifier
# Classification Metrics for RandomForestClassifier(max_depth=10, n_estimators=500, random_state=666)
# Accuracy: 1.0
# F1_Score: 1.0
# Precision: 1.0
# Recall: 1.0
# ROC AUC: 1.0
```

### Model saving (General Model)

```python
import pandas as pd
from fluid_domain.art import KMEngine
# Reading a dataset using pandas.
df=pd.read_csv('tested.csv')
engine=KMEngine(df,'Survived')
engine.model_save('Random Forest Classifier')

# This will save the specified model as pickle file in your wroking directory.
```

### Model saving (Custom Model)

The specified custom model will overwrite the existing model in the model dictionary, for space conservation purposes.

```python
import pandas as pd
from fluid_domain.art import KMEngine
# Reading a dataset using pandas.
df=pd.read_csv('tested.csv')
engine=KMEngine(df,'Survived')
custom_model = engine.set_custom_params('Random Forest Classifier', n_estimators=500, max_depth=10, random_state=666)
print(custom_model)
engine.model_save('Random Forest Classifier')

# This will save the updated specified model as pickle file in your wroking directory.
```

## Resuing the saved model.

```python
import pandas as pd
import pickle
saved_model=pickle.load(open('Random Forest Regression.pkl','rb'))
new_data = {
    'Make': ['ACURA','ACURA'],
    'Model': ['MDX 4WD','ILX'],
    'Vehicle Class': ['SUV - SMALL','COMPACT'],
    'Engine Size(L)': [3.5,5],
    'Cylinders': [6,12],
    'Transmission': ['AS6','AM7'],
    'Fuel Type': ['Z','D'],
    'Fuel Consumption City (L/100 km)': [11.2,13.5],
    'Fuel Consumption Hwy (L/100 km)': [10.0, 15.9],
    'Fuel Consumption Comb (L/100 km)': [25.36, 35.8],
    'Fuel Consumption Comb (mpg)': [40, 60]
}

# Convert the dictionary to a pandas DataFrame
new_data_df = pd.DataFrame(new_data)

# Use the saved model to make predictions on the new data
ypred = saved_model.predict(new_data_df)

print(ypred)
```
