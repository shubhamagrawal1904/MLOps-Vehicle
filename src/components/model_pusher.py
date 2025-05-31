import sys
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import    Proj1Estimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact:ModelEvaluationArtifact, model_pusher_config:ModelPusherConfig):
        
        self.s3=SimpleStorageService()
        self.model_evaluation_artifact=model_evaluation_artifact
        self.model_pusher_config=model_pusher_config
        self.proj1_estimator=Proj1Estimator(bucket_name=model_pusher_config.bucket_name, model_path=model_pusher_config.s3_model_key_path)
        
        
    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            print("--"*30)
            logging.info('Uploading new model to S3')
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact=ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                      s3_model_path=self.model_pusher_config.s3_model_key_path)
            
            logging.info('Uploaded Model to S3') 
            logging.info(f'Model Pusher Artifact : {model_pusher_artifact}')
            
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys)
        
        
        
        