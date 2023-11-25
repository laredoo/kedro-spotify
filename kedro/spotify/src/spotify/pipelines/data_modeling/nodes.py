import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    X = data.drop('Popularity', axis='columns')
    y = data['Popularity']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters['test_size']
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple:
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(X_train.values)
    y_train = scaler_y.fit_transform(y_train.values[:, np.newaxis])
    return X_train, y_train, scaler_x, scaler_y

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict) -> Ridge:
    regressor = GridSearchCV(Ridge(fit_intercept=False),
                         cv=parameters['cv'],
                         refit=parameters['refit'],
                         param_grid=parameters['param_grid'])
    regressor.fit(X_train, y_train)
    return regressor

def scaler_test(X_test: pd.DataFrame, y_test: pd.DataFrame, 
                scaler_x: StandardScaler, scaler_y: StandardScaler) -> Tuple:
    X_test = scaler_x.transform(X_test.values)
    y_test = scaler_y.transform(y_test.values[:, np.newaxis])
    return X_test, y_test

def evaluate_model(regressor: Ridge, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info('Model has a coefficient R^2 of %.3f and MSE of %.3f on test data', r2, mse)
