import os, sys
import pickle # saving model
import numpy as np 
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) # extract the directory path from a file path.
        os.makedirs(dir_path, exist_ok=True) # making directory

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info('Expection occoured in Pickling of Model')
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        model_report = {}
        model_list = list(models.values())
        model_name_list = list(models.keys())
        for i in range(0, len(models)):
            model = model_list[i]

            model.fit(X_train,y_train) 
            y_pred =model.predict(X_test)
            score = r2_score(y_test,y_pred)

            model_report[model_name_list[i]] = score

        return model_report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as obj:
            return pickle.load(obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function Utils')
        raise CustomException(e,sys)