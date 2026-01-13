import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load and preprocess data
heart_df = pd.read_csv('heart.csv')
heart_df = heart_df.drop(['oldpeak','slp','thall'], axis=1)
X = heart_df.drop('output', axis=1)
y = heart_df['output']

# Feature lists
numeric_features = ['age', 'trtbps', 'chol', 'thalachh']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'caa']

# Ensure all possible categories are present for OneHotEncoder
for col in categorical_features:
    if col == 'sex':
        X[col] = pd.Categorical(X[col], categories=[0,1])
    elif col == 'cp':
        X[col] = pd.Categorical(X[col], categories=[0,1,2,3])
    elif col == 'fbs':
        X[col] = pd.Categorical(X[col], categories=[0,1])
    elif col == 'restecg':
        X[col] = pd.Categorical(X[col], categories=[0,1,2])
    elif col == 'exng':
        X[col] = pd.Categorical(X[col], categories=[0,1])
    elif col == 'caa':
        X[col] = pd.Categorical(X[col], categories=[0,1,2,3,4])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
])

# Model pipelines
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True))
])
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Parameter grids
knn_param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'classifier__p': [1, 2]
}
svm_param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}
dt_param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

models = [
    ('KNN', knn_pipeline, knn_param_grid),
    ('SVM', svm_pipeline, svm_param_grid),
    ('DT', dt_pipeline, dt_param_grid)
]

best_model = None
best_score = 0
best_name = None

for name, pipeline, param_grid in models:
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    score = grid_search.best_score_
    if score > best_score:
        best_score = score
        best_model = grid_search.best_estimator_
        best_name = name

# Save best model pipeline to heart.pkl
with open('heart.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f'Best model ({best_name}) saved to heart.pkl with CV accuracy: {best_score:.4f}')
