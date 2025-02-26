import os 
import sys 
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score 


def evaluate_models(X_train, X_test, y_train, y_test, models):
    
    model_list = {}

    for name, model in models.items():
        # train the model
        model.fit(X_train, y_train)

        # predict the model
        y_pred = model.predict(X_test)

        # test model score
        test_model_score = r2_score(y_test, y_pred)
        
        # keep model name and score
        model_list[name] = test_model_score

    return model_list


def save_object(file_path, model):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(model, file_obj)

    except CustomException as e:
        raise CustomException(e, sys)