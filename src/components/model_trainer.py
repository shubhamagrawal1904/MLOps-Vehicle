import sys
import os

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact, DataTransformationArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact, model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config
        
    def get_model_object_report(self, train:np.array, test:np.array)->Tuple[object,object]:
        try:
            logging.info('Training RandomForest Classifier with Specified Parameters')
            x_train, y_train, x_test, y_test=train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info('train_test split done on transformed data')
            model=RandomForestClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                criterion=self.model_trainer_config._criterion,
                random_state=self.model_trainer_config._random_state
            )
            
            logging.info('Model Training going on.....')
            model.fit(x_train, y_train)
            logging.info('Model Training Done')
            
            
            y_pred=model.predict(x_test)
            accuracy=accuracy_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred)
            recall=recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            
            metric_artifact=ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            return model, metric_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
        
    def initiate_model_trainer(self)-> ModelTrainerArtifact:
        logging.info('Initiate Model Trainer Method of Model Trainer Class')
        try:
            train_arr=load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            trained_model, model_artifact=self.get_model_object_report(train=train_arr, test=test_arr)
            logging.info('Trained Model and Model Artifact Loaded')
            
            preprocessing_object=load_object(file_path=self.data_transformation_artifact.transformed_data_object_file_path)
            if accuracy_score(train_arr[:,-1], trained_model.predict(train_arr[:,:-1]))<self.model_trainer_config.expected_accuracy:
                logging.info('No Model found with expected accuracy')
                raise MyException('No Model found with above base score')
            logging.info('Saving Model as performance is better than previous model')
            mymodel=MyModel(preprocessing_object=preprocessing_object, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path,mymodel )
            
            
            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=ModelTrainerConfig.trained_model_file_path,
                                                        metric_artifact=model_artifact)
            logging.info('Model Atrifact Saved')
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys) from e