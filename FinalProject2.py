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
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'K-Nearest Neighbors Regressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [2, 4, 8, 16],
            'weights': ['uniform', 'distance'],
            'leaf_size': [10,30,50]
        }
    },
    'Support Vector Regressor': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'epsilon': [0.1, 0.2],
            'degree': [2, 3, 4]
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
        scoring='neg_mean_absolute_error',
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
    test_rmse = math.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
    train_r2 = r2_score(y_train_original, y_train_pred_original)
    test_r2 = r2_score(y_test_original, y_test_pred_original)
    
    results.update({model_name:{"best_Params":grid_search.best_params_,
                                "train_rmse":train_rmse,
                                "train_r2":train_r2,
                                "test_rmse":test_rmse,
                                "test_r2":test_r2,
                                "y_test":y_test_original,
                                "y_test_pred":y_test_pred_original}})
    
plt.style.use('ggplot')

for model in results:
    fig_size = (10, 8)

    plt.figure(figsize=fig_size)
    plt.scatter(results[model]["y_test"], results[model]["y_test_pred"], 
                alpha=0.7, s=60, color='steelblue', edgecolors='navy', linewidth=0.5)
    min_val = min(min(results[model]["y_test"]), min(results[model]["y_test_pred"]))
    max_val = max(max(results[model]["y_test"]), max(results[model]["y_test_pred"]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("True Charges", fontsize=12)
    plt.ylabel("Predicted Charges", fontsize=12)
    plt.title(f"Predicted vs True Charges - {model}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Residual Plot (Test)
    residuals = results[model]["y_test"] - results[model]["y_test_pred"]
    plt.figure(figsize=fig_size)
    plt.scatter(results[model]["y_test_pred"], residuals, 
                alpha=0.7, s=60, color='darkorange', edgecolors='brown', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Predicted Charges (Test)", fontsize=12)
    plt.ylabel("Residuals (Test)", fontsize=12)
    plt.title(f"Residuals vs Predicted Charges (Test) - {model}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    #Histogram of Residuals (Test)
    plt.figure(figsize=fig_size)
    plt.hist(residuals, bins=50, density=True, color='lightgreen', 
             edgecolor='darkgreen', linewidth=1.2, alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Residuals (Test)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Distribution of Residuals (Test) - {model}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
