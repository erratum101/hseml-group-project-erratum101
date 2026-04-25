# modeling.py
"""
Модуль для обучения и оценки моделей
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_xgboost(X_train, y_train, random_state=42):
    model = xgb.XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, random_state=42):
    model = lgb.LGBMRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def ensemble_predict(*preds):
    return np.mean(preds, axis=0)

def grid_search_rf(X_train, y_train):
    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
    gs = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_root_mean_squared_error')
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, -gs.best_score_
