import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from  src.utils import _save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path =os.path.join("artifact","model.pkl")
class ModelTraining:
    def __init__(self):
        self.model_trainer_congif =  ModelTrainingConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Linear Regression":LinearRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN":KNeighborsRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Ada Boost":AdaBoostRegressor(),
                "XGBoost":XGBRegressor()
            }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)        
            logging.info("model report generated")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise CustomException("Model score is very less")
            logging.info(f"Best found model on both training and testing dataset{best_model_name}")

            _save_object(
                file_path=self.model_trainer_congif.trained_model_file_path,
                obj=best_model

            )
            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)

