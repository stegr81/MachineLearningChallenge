# MachineLearningChallenge
A Python script to answer a coding challenge - use Machine Learning to predict the likelihood that breast cancer will reoccur in individuals provided within a dataset.

- Conducted initial exploration of the data and reported my findings.
- Cleaned the data and dealt with any irregularities.
- Once the data was prepared the script using Random Forest to attempt a prediction, using cross validation to get an average result.
- I used randomized search cross validation to get the best parameters in an attempt to improve the score.
- Attempted again using a different type of encoding.
- Finally I created a basic Artificial Neural Network using Tensor Flow to analyse the information.

Libraries required to run script: Some were defined as part of the challenge. Additionally certain things were attempted as mandated by the challenge. I've added notes to show my thinking and how I tried different things that specifically prescribed by the parameters of the challenge.


# Fixed dependencies
- pytest
- pandas
- numpy
- from google.colab import drive

# Preprocessing
- from sklearn.model_selection import train_test_split
- from sklearn.pipeline import Pipeline
- from sklearn.impute import SimpleImputer
- from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
- from sklearn.compose import ColumnTransformer

# Models
- from sklearn.ensemble import RandomForestClassifier

# Tuning
- from sklearn.model_selection import cross_val_score
- from sklearn.model_selection import RandomizedSearchCV

# Evaluating
- from sklearn.metrics import confusion_matrix, accuracy_score

# Neural Network
- tensorflow
- from tensorflow.keras.callbacks import LearningRateScheduler

