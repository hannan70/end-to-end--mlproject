import os
import sys 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_models(X_train, X_test, y_train, y_test, models,param=params)


            # get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            
            for name, score in model_report.items():
                if best_model_score == score:
                    best_model_name = name 

            # like `LinearRegression()` this is the best model for this dataset 
            best_model = models[best_model_name]

            # print(model_report)
            # print(best_model)

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




 