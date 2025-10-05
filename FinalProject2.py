import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Load data
Data = pd.read_csv("insurance.csv")

# One-hot encode
Data['sex'] = Data['sex'].map({'male': 0, 'female': 1})
Data['smoker'] = Data['smoker'].map({'no': 0, 'yes': 1})
Data['region'] = Data['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast':3})

# Log-transform charges
Data['charges'] = Data['charges'].apply(math.log1p)

# Normalize selected features
scaler = StandardScaler()
Data['age'] = scaler.fit_transform(Data[['age']])
Data['bmi'] = scaler.fit_transform(Data[['bmi']])
Data['children'] = scaler.fit_transform(Data[['children']])

X = Data.drop(columns=['charges'])
y = Data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15)

models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False]
        }
    },
    'Decision Tree Regressor': {
        'model': DecisionTreeRegressor(random_state=16),
        'params': {
            'max_depth': [2, 4, 8, 16],
            'min_samples_split': [2, 5, 10]
        }
    },
    'K-Nearest Neighbors Regressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [2, 4, 8, 16],
            'weights': ['uniform', 'distance']
        }
    },
    'Support Vector Regressor': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'epsilon': [0.1, 0.2],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
    }
}

# Train and evaluate
results = {}
for model_name, config in models.items():
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=10,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    print(model_name,config)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # transform back to original
    y_train_original = np.expm1(y_train)
    y_train_pred_original = np.expm1(y_train_pred)
    y_test_original = np.expm1(y_test)
    y_test_pred_original = np.expm1(y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
    train_r2 = r2_score(y_train_original, y_train_pred_original)
    test_r2 = r2_score(y_test_original, y_test_pred_original)
    
    print('best_model:', best_model)
    print('train_rmse:', train_rmse)
    print('test_rmse:', test_rmse)
    print('train_r2:', train_r2)
    print('test_r2:',test_r2)
    
# Example scatter plot of predictions vs true values (test set)
plt.scatter(y_test_original, y_test_pred_original, alpha=0.6)
plt.xlabel("True Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs True (Back-Transformed)")
plt.show()