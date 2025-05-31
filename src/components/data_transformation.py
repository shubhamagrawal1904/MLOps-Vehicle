import sys
import os
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransformation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, 
                 data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
            self._shema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            df=pd.read_csv(file_path)
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def get_data_transformer_object(self)->Pipeline:
        
        logging.info('Entered get_data_transformer_object method of DataTransformer Class')
        try:
            numeric_transformer=StandardScaler()
            min_max_scaler=MinMaxScaler()
            logging.info('Transformers Intialized : Standard Scaler - MinMax Scaler')
            
            num_features=self._shema_config['num_features']
            mm_columns=self._shema_config['mm_columns']
            
            preprocessor=ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxSclaer",min_max_scaler, mm_columns)
                ],
                remainder='passthrough'
            )
            
            final_pipeline=Pipeline(steps=[('Preprocessor',preprocessor)])
            logging.info('Final Pipeline Ready')
            logging.info('Exited get_data_transformer_object method of DataTransformer Class')
            return final_pipeline
        except Exception as e:
            raise MyException(e,sys) from e
        
    def _get_gender_column(self, df):
        logging.info('Mapping Gender Column to Binary Values')
        df['Gender']=df['Gender'].map({'Female':0, 'Male':1}).astype(int)
        return df
    
    def _create_dummy_cols(self,df):
        df=pd.get_dummies(df, drop_first=True)
        return df
    
    def _rename_cols(self, df):
        df=df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):
        logging.info('Dropping ID Column if it exists')
        drop_col=self._shema_config['drop_columns']
        if drop_col in df.columns:
            df=df.drop(drop_col, axis=1)
        return df
    
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        
        try:
            logging.info('Data Transformation Started!!!')
            
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df, test_df=(DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                            DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            logging.info('Test and Train Data Loaded')
            # logging.info(f'Train Data file path \n {self.data_ingestion_artifact.trained_file_path}')
            # print(f'Train Data : {train_df}')
            
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]
            
            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]
            
            
            input_feature_train_df=self._get_gender_column(input_feature_train_df)
            input_feature_train_df=self._drop_id_column(input_feature_train_df)
            input_feature_train_df=self._create_dummy_cols(input_feature_train_df)
            input_feature_train_df=self._rename_cols(input_feature_train_df)
            
            
            input_feature_test_df=self._get_gender_column(input_feature_test_df)
            input_feature_test_df=self._drop_id_column(input_feature_test_df)
            input_feature_test_df=self._create_dummy_cols(input_feature_test_df)
            input_feature_test_df=self._rename_cols(input_feature_test_df)
            
            logging.info('Custom transformations applied to train and test data')
            
            logging.info('Starting Data Transformation')
            preprocessor=self.get_data_transformer_object()
            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor.transform(input_feature_test_df)
            logging.info('Transformation done end to end')
            
            smt=SMOTEENN(sampling_strategy='minority')
            input_feature_train_final, target_feature_train_final=smt.fit_resample(input_feature_train_arr, target_feature_train_df)
            input_feature_test_final, target_feature_test_final=smt.fit_resample(input_feature_test_arr, target_feature_test_df)
            
            logging.info('SMOTEEN applied to train-test df')
            
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info(f"Training Data shape : {train_arr.shape}")
            logging.info("feature-target concatenation done for train-test df.")
            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            logging.info('Saving Transformed Data and object')
            
            logging.info('Transformation Completed!!!!')
            
            return DataTransformationArtifact(
                transformed_data_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys) from e
            
            
            
