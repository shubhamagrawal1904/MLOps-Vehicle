import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)
        
    def export_data_into_feature_store(self)->DataFrame:
        try:
            logging.info("Exporting Data from MongoDB")
            my_data=Proj1Data()
            dataframe=my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f'Saving Exported Data into feature store file path:{feature_store_file_path}')
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise MyException(e, sys)
        
    def split_data_as_train_test(self, dataframe:DataFrame)->DataFrame:
        try:
            train_set, test_set=train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info('Performed Train Test split on Dataframe')
            logging.info('Exited Train Test Split Module of Data Ingestion Class')
            dir_path=os.path.dirname(self.data_ingestion_config.trainging_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info('Exporting Train Test Split to file path')
            train_set.to_csv(self.data_ingestion_config.trainging_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info('Train Test Data Saved')
        except Exception as e:
            raise MyException(e, sys) from e
        
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        
        try: 
            dataframe=self.export_data_into_feature_store()
            logging.info('Got Data from MangoDB')
            self.split_data_as_train_test(dataframe=dataframe)
            logging.info('Performed train test split')
            logging.info('Exited Intitiate Data Ingestion Class')
            
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.trainging_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f'Data Ingestion Artifact : {data_ingestion_artifact}')
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e
            
            