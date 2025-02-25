import os 
import sys 
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill



def save_object(file_path, model):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(model, file_obj)

    except CustomException as e:
        raise CustomException(e, sys)