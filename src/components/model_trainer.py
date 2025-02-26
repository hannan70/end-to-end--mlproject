import os
import sys 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score 
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1], # X_train
                test_arr[:,:-1], # X_test
                train_arr[:, -1], # y_train
                test_arr[:, -1] # y_test
                
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict = evaluate_models(X_train, X_test, y_train, y_test, models)

            # get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            
            for name, score in model_report.items():
                if best_model_score == score:
                    best_model_name = name 

            # like `LinearRegression()` this is the best model for this dataset 
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                model=best_model
            )

            y_pred=best_model.predict(X_test)

            r2_square = r2_score(y_test, y_pred)

            return r2_square


        except CustomException as e:
            raise CustomException(e, sys)




 