import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig


from src.components.model_training import ModelTrainingConfig
from src.components.model_training import ModelTraining

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifact','train.csv')
    test_data_path:str = os.path.join('artifact','test.csv')
    raw_data_path:str = os.path.join('artifact','raw.csv')
class DataIngenstion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion method or components')
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info('Read the dataset as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train Test Split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=65)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    di=DataIngenstion()
    train_data,test_data=di.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTraining()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))